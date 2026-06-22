import os
import io
import base64
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from einops import rearrange
from flask import Flask, render_template, request, jsonify, send_file
from diffusers import AutoencoderKL, DDIMScheduler

app = Flask(__name__)

# Standard CelebA attribute keys in alphabetical order (matches training dataset)
ATTR_KEYS = [
    '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald',
    'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry',
    'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses',
    'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male',
    'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face',
    'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns',
    'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat',
    'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young',
]

CONFIG = {
    "dit_params": {
        "patch_size": 2,
        "num_layers": 24,
        "hidden_size": 768,
        "num_heads": 12,
        "head_dim": 64,
        "timestep_embed_dim": 768,
    },
    "autoencoder_params": {"z_channels": 4},
    "diffusion_params": {"num_steps": 1000, "beta_start": 1e-4, "beta_end": 0.02},
    "dataset_params": {"im_size": 256},
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ─── Model Definitions ────────────────────────────────────────────────────────

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


# ─── Noise Scheduler (DDPM) ──────────────────────────────────────────────────

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


# ─── Global Model State ───────────────────────────────────────────────────────

dit_model = None
vae_model = None
last_image_bytes = None


def load_models():
    global dit_model, vae_model
    cfg = CONFIG
    dit_img_size = cfg['dataset_params']['im_size'] // 8

    dit_model = DIT(
        dit_img_size,
        cfg['autoencoder_params']['z_channels'],
        cfg['dit_params'],
        attribute_num=len(ATTR_KEYS),
    )

    # Priority 1: local checkpoint folder
    ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoint')
    ckpt_files = sorted(glob.glob(os.path.join(ckpt_dir, '*.pth')))

    loaded = False
    if ckpt_files:
        # Pick highest epoch (files are named dit_model_epoch_N.pth)
        path = max(ckpt_files, key=lambda p: int(''.join(filter(str.isdigit, os.path.basename(p))) or '0'))
        print(f"Loading DiT from local checkpoint: {path}")
        ckpt = torch.load(path, map_location=device, weights_only=False)
        dit_model.load_state_dict(ckpt['model_state_dict'])
        loaded = True

    # Priority 2: HuggingFace
    if not loaded:
        print("No local checkpoint found. Downloading from HuggingFace wanderly0501/conditional-face-generator ...")
        try:
            from huggingface_hub import list_repo_files, hf_hub_download
            repo = "wanderly0501/conditional-face-generator"
            pth_files = [f for f in list_repo_files(repo) if f.endswith('.pth')]
            if not pth_files:
                raise RuntimeError("No .pth files found in HuggingFace repo")
            path = hf_hub_download(repo, pth_files[0])
            ckpt = torch.load(path, map_location=device, weights_only=False)
            state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
            dit_model.load_state_dict(state)
            loaded = True
            print(f"Loaded from HuggingFace: {pth_files[0]}")
        except Exception as exc:
            raise RuntimeError(f"Could not load model from HuggingFace: {exc}") from exc

    dit_model.to(device).eval()
    print("DiT model ready.")

    print("Loading VAE (stabilityai/sd-vae-ft-mse)...")
    vae_model = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    vae_model.eval()
    for p in vae_model.parameters():
        p.requires_grad = False
    print("All models loaded. Ready.")


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/device')
def device_info():
    return jsonify({'device': str(device)})


@app.route('/generate', methods=['POST'])
def generate():
    global last_image_bytes

    data = request.json or {}
    attributes = data.get('attributes', [0.0] * len(ATTR_KEYS))
    method = data.get('method', 'ddim')          # 'ddim' or 'ddpm'
    if method == 'ddpm':
        num_steps = 1000                         # DDPM always uses full 1000 steps
    else:
        num_steps = max(10, min(200, int(data.get('num_steps', 50))))

    if len(attributes) != len(ATTR_KEYS):
        return jsonify({'error': f'Expected {len(ATTR_KEYS)} attributes, got {len(attributes)}'}), 400

    try:
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        diff_cfg = CONFIG['diffusion_params']
        ae_cfg = CONFIG['autoencoder_params']
        im_size = CONFIG['dataset_params']['im_size'] // 8

        latent = torch.randn((1, ae_cfg['z_channels'], im_size, im_size), device=device)
        attrs = torch.tensor(attributes, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            if method == 'ddim':
                ddim = DDIMScheduler(
                    num_train_timesteps=1000,
                    beta_schedule='linear',
                    beta_start=diff_cfg['beta_start'],
                    beta_end=diff_cfg['beta_end'],
                    clip_sample=False,
                )
                ddim.set_timesteps(num_steps, device=device)
                for t in ddim.timesteps:
                    if device.type == 'cuda':
                        with torch.amp.autocast('cuda'):
                            noise_pred = dit_model(latent, t.unsqueeze(0), attrs)
                    else:
                        noise_pred = dit_model(latent, t.unsqueeze(0), attrs)
                    latent = ddim.step(noise_pred, t, latent, return_dict=False)[0]

            else:  # ddpm
                scheduler = LinearNoiseScheduler(
                    num_steps, diff_cfg['beta_start'], diff_cfg['beta_end']
                )
                for i in range(num_steps - 1, -1, -1):
                    t_tensor = torch.tensor([i], device=device)
                    if device.type == 'cuda':
                        with torch.amp.autocast('cuda'):
                            noise_pred = dit_model(latent, t_tensor, attrs)
                    else:
                        noise_pred = dit_model(latent, t_tensor, attrs)
                    latent = scheduler.sample_prev_timestep(latent, noise_pred, i).detach()

            decoded = vae_model.decode(latent).sample

        decoded = torch.clamp(decoded, -1.0, 1.0).detach().cpu()
        decoded = (decoded + 1) / 2
        img_pil = torchvision.transforms.ToPILImage()(decoded.squeeze(0))

        buf = io.BytesIO()
        img_pil.save(buf, format='PNG')
        last_image_bytes = buf.getvalue()

        return jsonify({'success': True, 'image': base64.b64encode(last_image_bytes).decode()})

    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(exc)}), 500


@app.route('/download')
def download():
    if last_image_bytes is None:
        return jsonify({'error': 'No image generated yet'}), 404
    buf = io.BytesIO(last_image_bytes)
    buf.seek(0)
    return send_file(buf, mimetype='image/png', as_attachment=True, download_name='generated_face.png')


if __name__ == '__main__':
    load_models()
    app.run(host='0.0.0.0', port=5000, threaded=True)
