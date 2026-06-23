import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ─── Position / Time Embeddings ───────────────────────────────────────────────

def get_patch_position_embedding(pos_embed_dim, grid_size, dev):
    g_h, g_w = grid_size
    h = torch.arange(g_h, dtype=torch.float32, device=dev)
    w = torch.arange(g_w, dtype=torch.float32, device=dev)
    grid = torch.stack(torch.meshgrid(h, w, indexing='ij'), dim=0)
    g_h_p, g_w_p = grid[0].reshape(-1), grid[1].reshape(-1)
    factor = 10000 ** (torch.arange(pos_embed_dim // 4, dtype=torch.float32, device=dev) / (pos_embed_dim // 4))
    def sincos(g):
        e = g[:, None].repeat(1, pos_embed_dim // 4) / factor
        return torch.cat([torch.sin(e), torch.cos(e)], dim=-1)
    return torch.cat([sincos(g_w_p), sincos(g_h_p)], dim=-1)


def get_time_embedding(time_steps, time_dim):
    factor = 10000 ** (torch.arange(time_dim // 2, dtype=torch.float32, device=time_steps.device) / (time_dim // 2))
    t = time_steps[:, None].repeat(1, time_dim // 2) / factor
    return torch.cat([torch.sin(t), torch.cos(t)], dim=-1)


# ─── Patch & Attribute Embeddings ─────────────────────────────────────────────

class PatchEmbedding(nn.Module):
    def __init__(self, im_h, im_w, im_c, p_h, p_w, hidden_size):
        super().__init__()
        self.p_h, self.p_w = p_h, p_w
        self.im_h, self.im_w = im_h, im_w
        self.patch_embed = nn.Linear(im_c * p_h * p_w, hidden_size)
        nn.init.xavier_uniform_(self.patch_embed.weight)
        nn.init.constant_(self.patch_embed.bias, 0)

    def forward(self, x):
        out = rearrange(x, 'b c (nh ph) (nw pw) -> b (nh nw) (ph pw c)', ph=self.p_h, pw=self.p_w)
        out = self.patch_embed(out)
        pos = get_patch_position_embedding(out.shape[-1], (self.im_h // self.p_h, self.im_w // self.p_w), x.device)
        return out + pos


class AttributesEmbedding(nn.Module):
    def __init__(self, attribute_num, hidden_size, embed_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(attribute_num, hidden_size),
            nn.GELU(approximate='tanh'),
            nn.Linear(hidden_size, embed_size),
        )

    def forward(self, x):
        return self.mlp(x)


# ─── Attention ────────────────────────────────────────────────────────────────

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config['num_heads']
        self.head_dim = config['head_dim']
        self.att_dim = self.n_heads * self.head_dim
        hs = config['hidden_size']
        self.qkv_proj = nn.Linear(hs, 3 * self.att_dim)
        self.output_proj = nn.Linear(self.att_dim, hs)
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.constant_(self.qkv_proj.bias, 0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0)

    def forward(self, x):
        q, k, v = self.qkv_proj(x).split(self.att_dim, dim=-1)
        q = rearrange(q, 'b n (nh hd) -> b nh n hd', nh=self.n_heads, hd=self.head_dim)
        k = rearrange(k, 'b n (nh hd) -> b nh n hd', nh=self.n_heads, hd=self.head_dim)
        v = rearrange(v, 'b n (nh hd) -> b nh n hd', nh=self.n_heads, hd=self.head_dim)
        att = F.softmax(torch.matmul(q, k.transpose(-1, -2)) * (self.head_dim ** -0.5), dim=-1)
        out = rearrange(torch.matmul(att, v), 'b nh n hd -> b n (nh hd)')
        return self.output_proj(out)


class CrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config['num_heads']
        self.head_dim = config['head_dim']
        self.att_dim = self.n_heads * self.head_dim
        hs = config['hidden_size']
        self.q_proj = nn.Linear(hs, self.att_dim)
        self.kv_proj = nn.Linear(hs, 2 * self.att_dim)
        self.output_proj = nn.Linear(self.att_dim, hs)
        for proj in [self.q_proj, self.kv_proj, self.output_proj]:
            nn.init.xavier_uniform_(proj.weight)
            nn.init.constant_(proj.bias, 0)

    def forward(self, x, attributes):
        q = self.q_proj(x)
        k, v = self.kv_proj(attributes).split(self.att_dim, dim=-1)
        q = rearrange(q, 'b n (nh hd) -> b nh n hd', nh=self.n_heads, hd=self.head_dim)
        k = rearrange(k, 'b 1 (nh hd) -> b nh 1 hd', nh=self.n_heads, hd=self.head_dim)
        v = rearrange(v, 'b 1 (nh hd) -> b nh 1 hd', nh=self.n_heads, hd=self.head_dim)
        att = F.sigmoid(torch.matmul(q, k.transpose(-1, -2)) * (self.head_dim ** -0.5))
        out = rearrange(torch.matmul(att, v), 'b nh n hd -> b n (nh hd)')
        return self.output_proj(out)


# ─── Transformer Layer ────────────────────────────────────────────────────────

class TransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        hs = config['hidden_size']
        self.cross_norm = nn.LayerNorm(hs, elementwise_affine=False, eps=1e-6)
        self.att_norm = nn.LayerNorm(hs, elementwise_affine=False, eps=1e-6)
        self.ff_norm = nn.LayerNorm(hs, elementwise_affine=False, eps=1e-6)
        self.attn_block = Attention(config)
        self.cross_block = CrossAttention(config)
        self.mlp_block = nn.Sequential(
            nn.Linear(hs, 4 * hs),
            nn.GELU(approximate='tanh'),
            nn.Linear(4 * hs, hs),
        )
        self.adaptive_norm_mlp = nn.Sequential(nn.SiLU(), nn.Linear(hs, 6 * hs))
        for lin in [self.mlp_block[0], self.mlp_block[-1]]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.constant_(lin.bias, 0)
        nn.init.xavier_uniform_(self.adaptive_norm_mlp[-1].weight)
        nn.init.constant_(self.adaptive_norm_mlp[-1].bias, 0)

    def forward(self, x, condition, attributes):
        ps, pa, po, ms, ma, mo = self.adaptive_norm_mlp(condition).chunk(6, dim=-1)
        out = x + self.cross_block(self.cross_norm(x), attributes)
        out = out + self.attn_block(self.att_norm(out) * (1 + pa.unsqueeze(1)) + ps.unsqueeze(1)) * po.unsqueeze(1)
        out = out + self.mlp_block(self.ff_norm(out) * (1 + ma.unsqueeze(1)) + ms.unsqueeze(1)) * mo.unsqueeze(1)
        return out


# ─── DiT ──────────────────────────────────────────────────────────────────────

class DIT(nn.Module):
    def __init__(self, im_size, im_channels, config, attribute_num=40):
        super().__init__()
        hs = config['hidden_size']
        self.p_h = self.p_w = config['patch_size']
        self.nh = self.nw = im_size // self.p_h
        self.im_channels = im_channels
        self.time_dim = config['timestep_embed_dim']
        self.patch_embed_layer = PatchEmbedding(im_size, im_size, im_channels, self.p_h, self.p_w, hs)
        self.attributes_embed_layer = AttributesEmbedding(attribute_num, 4 * hs, hs)
        self.t_proj = nn.Sequential(nn.Linear(self.time_dim, hs), nn.SiLU(), nn.Linear(hs, hs))
        self.layers = nn.ModuleList([TransformerLayer(config) for _ in range(config['num_layers'])])
        self.norm = nn.LayerNorm(hs, elementwise_affine=False, eps=1e-6)
        self.adaptive_norm_mlp = nn.Sequential(nn.SiLU(), nn.Linear(hs, 2 * hs))
        self.proj_out = nn.Linear(hs, self.p_h * self.p_w * im_channels)
        nn.init.normal_(self.t_proj[0].weight, std=0.02)
        nn.init.constant_(self.t_proj[0].bias, 0)
        for lin in [self.adaptive_norm_mlp[-1], self.proj_out]:
            nn.init.constant_(lin.weight, 0)
            nn.init.constant_(lin.bias, 0)

    def forward(self, img, t, attr):
        out = self.patch_embed_layer(img)
        t_embed = self.t_proj(get_time_embedding(torch.as_tensor(t).long(), self.time_dim))
        attri_token = self.attributes_embed_layer(attr).unsqueeze(1)
        for layer in self.layers:
            out = layer(out, t_embed, attri_token)
        shift, scale = self.adaptive_norm_mlp(out).chunk(2, dim=-1)
        out = self.norm(out) * (1 + scale) + shift
        out = self.proj_out(out)
        return rearrange(out, 'b (nh nw) (ph pw c) -> b c (nh ph) (nw pw)',
                         ph=self.p_h, pw=self.p_w, c=self.im_channels, nw=self.nw, nh=self.nh)


# ─── Noise Scheduler (DDPM) ───────────────────────────────────────────────────

class LinearNoiseScheduler:
    def __init__(self, num_timesteps, beta_start, beta_end):
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)

    def sample_prev_timestep(self, xt, pred, t):
        x0 = (xt - pred * self.sqrt_one_minus_alphas_cumprod.to(xt.device)[t]) \
             / self.sqrt_alphas_cumprod.to(xt.device)[t]
        x0 = torch.clamp(x0, -1.0, 1.0)
        mean = xt - (self.betas.to(xt.device)[t] * pred) \
               / self.sqrt_one_minus_alphas_cumprod.to(xt.device)[t]
        mean = mean / torch.sqrt(self.alphas.to(xt.device)[t])
        if t == 0:
            return mean
        variance = self.betas.to(xt.device)[t] \
                   * (1.0 - self.alphas_cumprod.to(xt.device)[t - 1]) \
                   / (1.0 - self.alphas_cumprod.to(xt.device)[t])
        return mean + torch.randn_like(x0) * (variance ** 0.5)
