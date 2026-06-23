import os
import io
import base64
import glob
import torch
import torchvision
from flask import Flask, render_template, request, jsonify, send_file
from diffusers import AutoencoderKL, DDIMScheduler

from model import DIT, LinearNoiseScheduler

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
