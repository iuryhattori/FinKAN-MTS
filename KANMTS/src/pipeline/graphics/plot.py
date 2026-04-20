import argparse
import os
import numpy as np
import torch
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
import matplotlib.pyplot as plt


def format_horizon(horizon, base_minutes=15):
    """
    Formata horizon de forma clara e legível.
    Suporta minutos, horas e dias.
    
    Args:
        horizon: índice do horizon (0, 1, 2, ...)
        base_minutes: intervalo base em minutos (padrão: 15)
    
    Returns:
        String formatada
    """
    minutes = base_minutes + (horizon * base_minutes)
    
    # Caso: >= 1 dia (1440 minutos)
    if minutes >= 1440:
        days = minutes / 1440
        
        if days == int(days):
            # Número exato de dias
            return f"Horizon: {int(days)} dia(s)"
        else:
            # Dias + horas + minutos
            d = int(days)
            remaining_minutes = minutes - (d * 1440)
            h = remaining_minutes // 60
            m = remaining_minutes % 60
            
            if m == 0:
                if h == 0:
                    return f"Horizon: {d} dia(s)"
                else:
                    return f"Horizon: {d} dia(s) e {h}h"
            else:
                if h == 0:
                    return f"Horizon: {d} dia(s) e {m}min"
                else:
                    return f"Horizon: {d} dia(s), {h}h e {m}min"
    
    # Caso: >= 1 hora (60 minutos)
    elif minutes >= 60:
        hours = minutes / 60
        
        if hours == int(hours):
            # Número exato de horas
            return f"Horizon: {int(hours)}h"
        else:
            # Horas + minutos
            h = int(hours)
            m = int((hours - h) * 60)
            return f"Horizon: {h}h{m:02d}min" if m > 0 else f"Horizon: {h}h"
    
    # Caso: < 1 hora
    else:
        return f"Horizon: {minutes}min"


class PredictionPlotter:
    def __init__(self, 
                model_path : str,
                denormalize : bool = True,
                save_path : str = '.plots/'
                ):
        self.model_path = model_path
        self.denormalize = denormalize

        self._preds : np.ndarray | None = None
        self._trues : np.ndarray | None = None
        self.save_path = save_path

    def load_test(self, args : argparse.Namespace, device : str = 'cuda') -> None:
        exp = Exp_Long_Term_Forecast(args)

        exp.model.load_state_dict(torch.load(self.model_path, map_location=device))
        exp.model.to(device)
        exp.model.eval()

        test_data, test_loader = exp._get_data(flag='test')
        self._channels = test_data.get_channel_names()

        all_preds = []
        all_trues = []

        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                batch_x_mark = batch_x_mark.float().to(device)
                batch_y_mark = batch_y_mark.float().to(device)

                dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)

                outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs[:, -args.pred_len:, :].detach().cpu().numpy()
                true = batch_y[:, -args.pred_len:, :].detach().cpu().numpy()

                all_preds.append(pred)
                all_trues.append(true)

                if (i + 1) % 50 == 0:
                    print(f"   Processed {i+1}/{len(test_loader)} batches...")
        all_preds = np.concatenate(all_preds, axis=0)
        all_trues = np.concatenate(all_trues, axis=0)

        if self.denormalize:
            self._preds = test_data.inverse_transform(all_preds)
            self._trues = test_data.inverse_transform(all_trues)
        else:   
            self._preds = all_preds
            self._trues = all_trues

    def plot_feature(
            self,
            feature_name : str,
            horizon      : int = 0,
            fig_size     : tuple = (20, 6),
            show_plot    : bool  = False,
    ):

        feature_idx = self._channels.index(feature_name)
        preds_flat = self._preds[:, horizon, feature_idx]
        trues_flat = self._trues[:, horizon, feature_idx]
        metrics = self._compute_metrics(preds_flat, trues_flat)
        self._render_and_save(preds_flat, trues_flat, horizon, feature_name, metrics, fig_size, show_plot)

    def plot_all_features(
            self,
            horizon : int = 0,
            figsize : tuple = (20, 6),
            show_plot: bool = False,
    ):            
        targets = self._channels
        print(f"Plotting {len(targets)} features - {format_horizon(horizon)}\n")
        for feat in targets:
            idx = targets.index(feat)
            metrics = self._compute_metrics(self._preds[:, horizon, idx], self._trues[:, horizon, idx])
            self._render_and_save(self._preds[:, horizon, idx], self._trues[:, horizon, idx], horizon, feat, metrics, figsize, show_plot)


        
    def _render_and_save(self,
                        preds,
                        truths,
                        horizon,
                        feature_name,
                        metrics,
                        figsize,
                        show_plot,):
        fig, ax = plt.subplots(figsize=figsize)
        times = np.arange(len(preds))

        ax.plot(times, truths, label="Actual", color='#2E86AB', linewidth=2, alpha=0.8)
        ax.plot(times, preds, label="Predicted", color= '#A23B72', linewidth=2, alpha=0.8)

        scale = '(Real Scale)' if self.denormalize else '(normalized)'
        ax.set_xlabel('Sample Index', fontsize=13, fontweight='bold')
        ax.set_ylabel(f'{feature_name} {scale}', fontsize=13, fontweight='bold')
        ax.set_title(
            f"{feature_name} | {format_horizon(horizon)}\n"
            f"MAE={metrics['mae']: .4f} | MSE={metrics['mse']: .4f}",
            fontsize=13, fontweight='bold', pad=15
        )
        ax.legend(fontsize=12, loc='upper left')
        ax.grid(True, alpha=0.3, linestyle='--')
        fig.tight_layout()

        os.makedirs(self.save_path, exist_ok=True)
        scale_s = "_real" if self.denormalize else "_norm"
        save_file = os.path.join(self.save_path, f"prediction_{feature_name}{scale_s}_h{horizon}.png")
        fig.savefig(save_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_file}")

        if show_plot:
            plt.show()
        plt.close(fig)

    @staticmethod
    def _compute_metrics(preds : np.ndarray, trues : np.ndarray) -> dict:
        mae = float(np.mean(np.abs(preds - trues)))
        mse = float(np.mean((preds - trues) ** 2))
        
        return {'mae': mae, 'mse': mse}


