from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import torch
from chronos import ChronosPipeline
from datetime import datetime, timedelta
import warnings
import traceback

warnings.filterwarnings("ignore")

app = FastAPI()

class VentaHistorica(BaseModel):
    fecha: datetime
    total: float

class PrediccionRequest(BaseModel):
    ventas: list[VentaHistorica]
    agrupacion: str = "dia"

pipeline: ChronosPipeline = None
device: str = "cpu"

@app.on_event("startup")
async def startup_event():
    global pipeline, device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n✅ Usando dispositivo para Chronos: {device}")
    try:
        pipeline = ChronosPipeline.from_pretrained(
            "amazon/chronos-t5-tiny",
            device_map=device,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
        )
        print("✅ Modelo cargado correctamente.")
    except Exception as e:
        print(f"❌ Error al cargar modelo: {e}")
        pipeline = None

@app.post("/predecir")
async def predecir_ventas(request_data: PrediccionRequest):
    print("🔥 Endpoint /predecir fue llamado.")
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Modelo de predicción no cargado.")

    try:
        agrupacion = request_data.agrupacion.lower()
        df_ventas = pd.DataFrame([v.model_dump() for v in request_data.ventas])
        df_ventas['fecha'] = pd.to_datetime(df_ventas['fecha'])
        df_ventas = df_ventas.sort_values('fecha')

        # 🔧 Rellenar fechas faltantes
        start_date = df_ventas['fecha'].min()
        end_date = df_ventas['fecha'].max()

        if agrupacion == "dia":
            freq = "D"
            prediction_length = 30
        elif agrupacion == "mes":
            freq = "MS"
            prediction_length = 12
        elif agrupacion == "año":
            freq = "YS"
            prediction_length = 4
        else:
            raise HTTPException(status_code=400, detail="Agrupación inválida")

        full_dates = pd.date_range(start=start_date, end=end_date, freq='D')
        df_completo = pd.DataFrame({'fecha': full_dates})
        df_agrupada = df_ventas.groupby('fecha').sum(numeric_only=True).reset_index()
        df_merged = pd.merge(df_completo, df_agrupada, on='fecha', how='left').fillna(0)

        # 🧮 Agrupar según tipo
        df_agrupada = df_merged.groupby(pd.Grouper(key='fecha', freq=freq)).sum(numeric_only=True)
        df_agrupada['total'] = pd.to_numeric(df_agrupada['total'], errors='coerce').fillna(0)

        values = df_agrupada['total'].values.astype(np.float32)

        if len(values) < 3:
            raise HTTPException(status_code=400, detail="Se requieren al menos 3 puntos para predecir.")

        # 🎯 Evaluación de precisión usando hold-out (solo si hay suficientes datos)
        try:
            if len(values) >= 10:
                split_idx = int(len(values) * 0.8)
                train, test = values[:split_idx], values[split_idx:]
                input_tensor_eval = torch.tensor(train, dtype=torch.float32, device=device)
                pred_eval = pipeline.predict([input_tensor_eval], prediction_length=len(test))[0]
                forecast_eval = pred_eval.median(dim=0)[0].cpu().numpy()
                mae = float(np.mean(np.abs(forecast_eval - test)))
                mape = float(np.mean(np.abs((forecast_eval - test) / (test + 1e-8))) * 100)
            else:
                mae, mape = None, None
        except Exception as e:
            print("⚠️ Error evaluando precisión:", e)
            mae, mape = None, None

        # 🔮 Predicción final
        print("✅ Valores agrupados para predicción:", values)

        input_tensor = torch.tensor(values, dtype=torch.float32, device=device)
        forecast_tensor = pipeline.predict([input_tensor], prediction_length=prediction_length)[0]

        forecast_median = forecast_tensor.median(dim=0)[0].cpu().numpy()
        forecast_lower = forecast_tensor.quantile(0.05, dim=0).cpu().numpy()
        forecast_upper = forecast_tensor.quantile(0.95, dim=0).cpu().numpy()

        last_date = df_agrupada.index[-1]
        if agrupacion == "dia":
            future_dates = [last_date + timedelta(days=i + 1) for i in range(prediction_length)]
        elif agrupacion == "mes":
            future_dates = [last_date + pd.DateOffset(months=i + 1) for i in range(prediction_length)]
        elif agrupacion == "año":
            future_dates = [last_date + pd.DateOffset(years=i + 1) for i in range(prediction_length)]

        prediccion_list = []
        for i in range(prediction_length):
            prediccion_list.append({
                "ds": future_dates[i].strftime('%Y-%m-%d'),
                "yhat": float(forecast_median[i]),
                "yhat_lower": float(forecast_lower[i]),
                "yhat_upper": float(forecast_upper[i]),
            })

        # 🔍 Tendencia
        mid = prediction_length // 2
        avg_start = np.mean(forecast_median[:mid])
        avg_end = np.mean(forecast_median[mid:])
        tendencia = "estable"
        if avg_end > avg_start * 1.05:
            tendencia = "positiva"
        elif avg_end < avg_start * 0.95:
            tendencia = "negativa"

        return {
            "prediccion": prediccion_list,
            "tendencia": tendencia,
            "mae": round(mae, 2) if mae is not None else None,
            "mape": round(mape, 2) if mape is not None else None
        }

    except Exception as e:
        print("❌ Error durante predicción:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
