import torch
from torch import nn
from torchvision.transforms import v2 as T
from torchvision import tv_tensors
from torchvision.io import read_image

import numpy as np
from tqdm import tqdm
from pathlib import Path

from choroidalyze.model import UNet
from choroidalyze.metrics import compute_measurement

from PIL import Image, ImageDraw,ImageFont
import numpy as np
import matplotlib.pyplot as plt
import os

import sys

if getattr(sys, 'frozen', False):
    # Executável PyInstaller
    base_path = sys._MEIPASS
else:
    # Código normal em .py
    base_path = os.path.abspath(".")

def get_default_img_transforms():
    return T.Compose([
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=(0.5,), std=(0.5,)),
    ])


class Choroidalyzer:
    DEFAULT_MODEL_URL = 'https://github.com/justinengelmann/Choroidalyzer/releases/download/' \
                        'v1.0/choroidalyzer_model_weights.pth'
    DEFAULT_SCALE = (11.49, 3.87)
    DEFAULT_THRESHOLDS = (0.5, 0.5, 0.1)

    def __init__(self, model_name='default', device='cpu',
                 default_scale=DEFAULT_SCALE, default_thresholds=DEFAULT_THRESHOLDS,
                 img_transforms=None, local_weights_path=None,
                 override_fovea_to_center=False, macula_rum=3000):
        self.model_name = model_name
        self.device = device
        self.default_scale = default_scale
        self.default_thresholds = default_thresholds
        self.img_transforms = img_transforms or get_default_img_transforms()
        self.local_weights_path = local_weights_path
        self.override_fovea_to_center = override_fovea_to_center
        self.macula_rum = macula_rum

        self._init_model()
        self.outputs = ['region', 'vessel', 'fovea']
        self.fovea_signal_filter = None

    def _init_model(self):
        assert self.model_name == 'default', 'Only default model is supported at this time'

        self.model = UNet(in_channels=1, out_channels=3, depth=7, channels='8_doublemax-64',
                          up_type='conv_then_interpolate', extra_out_conv=True)
        if self.local_weights_path:
            state_dict = torch.load(self.local_weights_path, map_location='cpu')
        else:
            state_dict = torch.hub.load_state_dict_from_url(self.DEFAULT_MODEL_URL, map_location='cpu')

        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model.to(self.device)

    def _load_image_if_needed(self, img_path_or_object: [str, Path, torch.Tensor, np.ndarray]) -> torch.Tensor: # type: ignore
        if isinstance(img_path_or_object, (str, Path)):
            img = read_image(str(img_path_or_object))
        elif isinstance(img_path_or_object, np.ndarray):
            img = torch.from_numpy(img_path_or_object)
        elif isinstance(img_path_or_object, torch.Tensor):
            img = img_path_or_object
        else:
            raise ValueError(f'Invalid input type for loading image. Must be str, Path, torch.Tensor, or np.ndarray'
                             f', but got {type(img_path_or_object)}. {img_path_or_object}')
        return img

    def _load_transform_image(self, img: str):
        img = self._load_image_if_needed(img)
        img = tv_tensors.Image(img)
        img = self.img_transforms(img)
        return img

    @torch.inference_mode()
    def analyze(self, img_path_or_object: [str, Path, torch.Tensor, np.ndarray], thresholds=None, scale=None) -> dict: # type: ignore
        thresholds = thresholds or self.default_thresholds
        scale = scale or self.default_scale

        preds = self.predict(img_path_or_object)

        region_mask = preds[0].ge(thresholds[0])
        vessel_mask = preds[1].ge(thresholds[1])
        region_mask = region_mask.cpu().numpy()
        vessel_mask = vessel_mask.cpu().numpy()
        
        if not self.override_fovea_to_center:
            fov_loc = self.process_fovea_prediction(preds.unsqueeze(0))
        else:
            fov_loc = None

        try:
            raw_thickness, area, vascular_index, choroid_vessel_area = compute_measurement(reg_mask=region_mask,
                                                                                           vess_mask=vessel_mask,
                                                                                           fovea=fov_loc,
                                                                                           macula_rum=self.macula_rum,
                                                                                           scale=scale)
        except:
            try:
                raw_thickness, area, vascular_index, choroid_vessel_area = compute_measurement(reg_mask=region_mask,
                                                                                           vess_mask=vessel_mask,
                                                                                           fovea=None,
                                                                                           macula_rum=self.macula_rum,
                                                                                           scale=scale)
            except ValueError as e:
                raise ValueError(f'Metrics calculation failed with the following error: {e}\nThis might be due to the fovea detection failing or the region of interest being too large.')
            
        thickness = np.mean(raw_thickness)
        return {'thickness': thickness, 'area': area, 'vascular_index': vascular_index,
                'vessel_area': choroid_vessel_area, 'raw_thickness': raw_thickness}

    @torch.no_grad()
    def process_fovea_prediction(self, preds):
        def _get_fov_filter(kernel_size=21):
            assert kernel_size % 2 == 1
            fov_filter = nn.Conv1d(1, 1, kernel_size=kernel_size, bias=False, padding_mode='reflect', padding='same')
            fov_filter.requires_grad_(False)
            ascending_weights = torch.linspace(0.1, 1, kernel_size // 2)
            fov_filter_weights = torch.cat([ascending_weights, torch.tensor([1.]), ascending_weights.flip(0)])
            fov_filter_weights /= fov_filter_weights.sum()
            fov_filter.weight = torch.nn.Parameter(fov_filter_weights.view(1, 1, -1), requires_grad=False)
            return fov_filter

        def _agg_fov_signal(tens, d=2):
            return tens[:, 2:, :].sum(dim=d)

        if self.fovea_signal_filter is None:
            self.fovea_signal_filter = (_get_fov_filter(kernel_size=21),
                                        _get_fov_filter(kernel_size=51))

        # we need d=2 (vert) and d=3 (horiz)
        out = []
        for d, filter in zip([2, 3], self.fovea_signal_filter):
            filter.to(preds.device)
            fov_signal = _agg_fov_signal(preds, d)
            fov_signal = filter(fov_signal).squeeze()
            out.append(fov_signal.argmax(dim=-1).item())

        return tuple(out)

    @torch.inference_mode()
    def predict(self, img_path: str):
        img = self._load_transform_image(img_path)
        img = img.to(self.device)
        with torch.no_grad():
            pred = self.model(img.unsqueeze(0)).sigmoid()
        return pred.squeeze(0).cpu()

    @torch.inference_mode()
    
    
    def predict_and_give_images(self, img_path_or_object, name, number, date, eye, metrics, exam_date, surgery_status, save_dir, thresholds=None):
        
        
        
        

        thresholds = thresholds or self.default_thresholds

        # 1. Predição
        preds = self.predict(img_path_or_object)
        fov_loc = self.process_fovea_prediction(preds.unsqueeze(0))

        if isinstance(thresholds, (int, float)):
            thresholds = (thresholds, thresholds, thresholds)
        preds = [_.ge(thresholds[i]) for i, _ in enumerate(preds)]

        # 2. Carregar imagem original
        img = self._load_image_if_needed(img_path_or_object)
        if img.ndim == 2:
            img = img.unsqueeze(0)
        elif img.ndim == 3 and img.shape[0] > 1:
            img = img[0:1, :, :]

        # 3. Criar diretório se necessário
        os.makedirs(save_dir, exist_ok=True)

        # 4. Texto informativo
        metadata = [
        f"Patient's name: {name}",
        f"Healthcare number: {number}",
        f"Eye: {eye}",
        f"Date of analysis: {date if date else '---'}",
        ]
        if exam_date:
            metadata.append(f"Date of acquisition: {exam_date}")
        if surgery_status:
            metadata.append(f"Intervention Status: {surgery_status}")

        metrics = [
            f"Thickness: {metrics['thickness']} µm",
            f"Area: {metrics['area']} mm²",
            f"Vascular Index: {metrics['vascular_index']}",
            f"Vessel Area: {metrics['vessel_area']} mm²"
        ]


        

        # 6. Original
        img_array = img.squeeze(0).cpu().numpy()

		# Normalize to 0-255 if needed
        if img_array.dtype != np.uint8:
            img_array = (img_array - img_array.min()) / (np.ptp(img_array))
            img_array = (img_array * 255).astype(np.uint8)

		# Convert to RGB so we can draw colored lines
        img_rgb = np.stack([img_array]*3, axis=-1)

		# Draw the fovea lines
        img_pil = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(img_pil)

        x = int(fov_loc[0])
        y = int(fov_loc[1])

		# Draw lines with thicker width
        draw.line([(x, 0), (x, img_array.shape[0])], fill=(255, 0, 0), width=1)
        draw.line([(0, y), (img_array.shape[1], y)], fill=(255, 0, 0), width=1)

		# Save: adiciona barra de escala à imagem JÁ com eixos vermelhos
        scale_x = self.default_scale[0]
        scale_y = self.default_scale[1]

        # Converte img_pil (com eixos) para numpy RGB array
        image_with_axes_array = np.array(img_pil)

        # Adiciona barra de escala à imagem com eixos e guarda como 'original.png'
        save_with_manual_scale_bar(image_with_axes_array, scale_x / 1000, scale_y / 1000,
                                os.path.join(save_dir, 'original.png'), bar_length_mm=0.2)

        # 7. Região
        plt.imsave(os.path.join(save_dir, 'region.png'), preds[0].cpu(), cmap='gray')

        # 8. Vasos
        plt.imsave(os.path.join(save_dir, 'vessel.png'), preds[1].cpu(), cmap='gray')

        # 9. Fóvea
        

        # Original fovea array
        fovea_array = preds[2].cpu().numpy()

        # Normalize to 0-255
        fovea_array = fovea_array.astype(np.uint8) * 255

        # Convert to RGB
        fovea_rgb = np.stack([fovea_array]*3, axis=-1)

        # Draw lines
        img = Image.fromarray(fovea_rgb)
        draw = ImageDraw.Draw(img)
        x = int(fov_loc[0])
        y = int(fov_loc[1])
        draw.line([(x, 0), (x, fovea_array.shape[0])], fill=(255, 0, 0), width=3)
        draw.line([(0, y), (fovea_array.shape[1], y)], fill=(255, 0, 0), width=3)

        # Save
        img.save(os.path.join(save_dir, 'fovea.png'))

        logo_tecnico = os.path.join(base_path,"Tecnico.png")  # Ajuste conforme necessário
        logo_fmul= os.path.join(base_path, "FMUL.png")
        # Adiciona também a barra de escala à imagem PIL com eixos antes de gerar relatório
        img_pil_with_scale = adicionar_barra_escala_pil(img_pil, scale_x / 1000, scale_y / 1000)

        # Passa imagem com escala para o relatório
        gerar_relatorio_em_memoria(save_dir, img_pil_with_scale, preds[0].cpu(), preds[1].cpu(), img, logo_tecnico, logo_fmul, metadata, metrics)



def save_with_manual_scale_bar(image_array, scale_x_mm_per_pixel, scale_y_mm_per_pixel, save_path, bar_length_mm=1.0):
    fig, ax = plt.subplots(figsize=(6, 6))

    # Verifica se é RGB ou grayscale
    if image_array.ndim == 2:  # Grayscale
        ax.imshow(image_array, cmap='gray')
        h, w = image_array.shape
    elif image_array.ndim == 3 and image_array.shape[2] == 3:  # RGB
        ax.imshow(image_array)
        h, w, _ = image_array.shape
    else:
        raise ValueError("Formato de imagem inválido. Esperado 2D (grayscale) ou 3D (RGB).")

    # Barra horizontal
    length_pixels_horizontal = int(bar_length_mm / scale_x_mm_per_pixel)
    x_start = 20
    y_start = h - 20
    ax.hlines(y=y_start, xmin=x_start, xmax=x_start + length_pixels_horizontal, color='white', linewidth=3)
    ax.text(x_start, y_start - 10, f'{bar_length_mm*1000:.0f} µm', color='white', fontsize=12)

    # Barra vertical
    length_pixels_vertical = int(bar_length_mm / scale_y_mm_per_pixel)
    ax.vlines(x=x_start, ymin=y_start - length_pixels_vertical, ymax=y_start, color='white', linewidth=3)

    ax.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def gerar_relatorio_em_memoria(save_dir, img_pil, region_tensor, vessel_tensor, fovea_img, logo_tecnico, logo_fmul, metadata, metrics):


    # Configurações gerais
    largura = 850
    altura = 1500
    relatorio = Image.new('RGB', (largura, altura), 'white')
    draw = ImageDraw.Draw(relatorio)

    # Fontes
    try:
        fonte_titulo = ImageFont.truetype("arialbd.ttf", 26)
        fonte_subtitulo = ImageFont.truetype("arialbd.ttf", 18)
        fonte_texto = ImageFont.truetype("arial.ttf", 14)
    except:
        fonte_titulo = ImageFont.load_default()
        fonte_subtitulo = ImageFont.load_default()
        fonte_texto = ImageFont.load_default()

    # Inserir logo (canto superior esquerdo e direito)
    logo_tecnico = Image.open(logo_tecnico).resize((142, 100))
    logo_fmul = Image.open(logo_fmul).resize((100, 100))
    relatorio.paste(logo_fmul, (75, 30))
    relatorio.paste(logo_tecnico, (630, 30))

    # Título
    draw.text((largura // 2 - 200, 40), "Report of Choroid Analysis", font=fonte_titulo, fill='black')

    # Headers (metadata e métricas)
    metadata_header = "PATIENT METADATA:"
    metrics_header = "CHOROID METRICS:"

    # Ajustar Y inicial
    y_info = 220

    # Metadata alinhado à esquerda
    draw.text((75, y_info), metadata_header, font=fonte_subtitulo, fill='black')
    meta_text = "\n".join(metadata)
    draw.text((75, y_info + 30), meta_text, font=fonte_texto, fill='black')

    # Metrics alinhado à direita
    draw.text((475, y_info), metrics_header, font=fonte_subtitulo, fill='black')
    metrics_text = "\n".join(metrics)
    draw.text((475, y_info + 30), metrics_text, font=fonte_texto, fill='black')

    # Converter imagens para 300x300 px
    def process_image(tensor_or_img):
        if isinstance(tensor_or_img, np.ndarray):
            arr = tensor_or_img
        else:
            arr = tensor_or_img.numpy()
        if arr.ndim == 2:
            arr = (arr * 255).astype(np.uint8)
            img = Image.fromarray(arr).convert('RGB')
        else:
            img = tensor_or_img.convert('RGB')
        return img.resize((300, 300))

    original_image = img_pil.resize((300, 300))
    region_image = process_image(region_tensor)
    vessel_image = process_image(vessel_tensor)
    fovea_image = fovea_img.resize((300, 300))

    # Labels e imagens
    labels = ["Original", "Region", "Vessel", "Fovea"]
    images = [original_image, region_image, vessel_image, fovea_image]

    # Espaços e margens
    x_margin = 75
    x_spacing = 100
    y_start = 400
    y_spacing = 80

    # Colocar imagens e rótulos
    for i, (label, img) in enumerate(zip(labels, images)):
        col = i % 2
        row = i // 2
        x = x_margin + col * (300 + x_spacing)
        y = y_start + row * (300 + y_spacing)

        # Rótulo centralizado
        bbox = fonte_subtitulo.getbbox(label)
        label_w = bbox[2] - bbox[0]
        label_x = x + (300 - label_w) // 2
        draw.text((label_x, y - 25), label, font=fonte_subtitulo, fill='black')
        relatorio.paste(img, (x, y))

    # Salvar como PDF
    relatorio_path = os.path.join(save_dir, 'report.pdf')
    relatorio.save(relatorio_path, "PDF", resolution=300, quality=95)
    print(f"Report saved in: {relatorio_path}")

def adicionar_barra_escala_pil(img_pil, scale_x_mm_per_pixel, scale_y_mm_per_pixel, bar_length_mm=0.2):
    """
    Adiciona barra de escala branca à imagem PIL (com eixos).
    """
    draw = ImageDraw.Draw(img_pil)
    w, h = img_pil.size

    # Converter comprimento em mm para píxeis
    length_px_x = int(bar_length_mm / scale_x_mm_per_pixel)
    length_px_y = int(bar_length_mm / scale_y_mm_per_pixel)

    x_start = 20
    y_start = h - 20

    # Linha horizontal
    draw.line((x_start, y_start, x_start + length_px_x, y_start), fill='white', width=3)

    # Texto
    font_size = 16
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    draw.text((x_start, y_start - 20), f"{int(bar_length_mm * 1000)} µm", fill='white', font=font)

    # Linha vertical
    draw.line((x_start, y_start, x_start, y_start - length_px_y), fill='white', width=3)

    return img_pil
