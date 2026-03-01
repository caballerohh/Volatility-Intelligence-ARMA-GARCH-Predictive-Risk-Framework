import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
import io
import warnings

# --- LIBRERÍAS PDF (ReportLab) ---
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas
from reportlab.platypus import Frame, Paragraph, Table, TableStyle
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader

# --- LIBRERÍAS ECONOMETRÍA ---
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import het_arch, acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")
plt.style.use('seaborn-v0_8-whitegrid')

# 1. CONFIGURACIÓN GLOBAL DEL CÓDIGO

TICKERS = ["AAPL", "AMZN", "JPM", "GS", "SPY", "DIA", "BND"]
PESOS = [0.1311, 0.1317, 0.1318, 0.1335, 0.1595, 0.1619, 0.1478]
FECHA_INICIO = "2015-01-01"
FECHA_FIN = "2025-11-27"

COLORS = {'returns': '#1f77b4', 'line0': 'black', 'Normal': '#D62728', 't-Student': '#FF7F0E', 'GED': '#9467BD'}

DESCRIPCIONES = {
    "AAPL": "<b>Apple Inc.:</b> Líder tecnológico en hardware y servicios.",
    "AMZN": "<b>Amazon:</b> E-commerce y Cloud (AWS).",
    "JPM": "<b>JPMorgan:</b> Banca Sistémica Global.",
    "GS": "<b>Goldman:</b> Banca de Inversión.",
    "SPY": "<b>S&P 500:</b> Benchmark de Mercado.",
    "DIA": "<b>Dow Jones:</b> Industriales Blue-Chip.",
    "BND": "<b>Total Bond:</b> Renta Fija.",
    "PORTAFOLIO": "<b>PORTAFOLIO:</b> Diversificación Estratégica."
}

NOTAS = {
    "AAPL": "Sensible Tech.", "AMZN": "Consumo Cíclico.",
    "JPM": "Ciclo Financiero.", "GS": "Alto Beta.", "SPY": "Referencia.",
    "DIA": "Value.", "BND": "Refugio.", "PORTAFOLIO": "Eficiente."
}

# 2. MOTORES DE CÁLCULO (GARCH Y PRUEBAS)

def get_stage1_stats(serie):
    """Estadísticas preliminares (ADF, Ljung-Box, ARCH-LM)"""
    try: adf_p = adfuller(serie.dropna())[1]
    except: adf_p = 1.0
    try: lb_p = acorr_ljungbox(serie.dropna(), lags=[10], return_df=True)['lb_pvalue'].iloc[0]
    except: lb_p = 1.0
    try: arch_p = het_arch(serie.dropna())[1]
    except: arch_p = 1.0
    return adf_p, lb_p, arch_p

def get_stage3_stats(var_series, retornos, alpha=0.05):
    """Pruebas de Backtesting (Kupiec y Christoffersen)"""
    hits = ((retornos < -var_series) * 1).values
    T = len(hits)
    N = np.sum(hits)
    N_exp = T * alpha

    # Kupiec (Unconditional Coverage)
    p_hat = N / T
    try:
        if N > 0 and N < T:
            num = (1 - alpha)**(T - N) * alpha**N
            den = (1 - p_hat)**(T - N) * p_hat**N
            lr_uc = -2 * np.log(num / den)
            kupiec_p = 1 - stats.chi2.cdf(lr_uc, 1)
        else: kupiec_p = 0.0
    except: kupiec_p = 0.0

    # Christoffersen (Conditional Coverage)
    try:
        tr = np.zeros((2, 2))
        for i in range(1, T): tr[hits[i-1], hits[i]] += 1
        n00, n01, n10, n11 = tr[0,0], tr[0,1], tr[1,0], tr[1,1]
        pi = (n01+n11)/T
        pi0 = n01/(n00+n01) if (n00+n01)>0 else 0
        pi1 = n11/(n10+n11) if (n10+n11)>0 else 0

        if pi>0 and pi<1:
            l_ind = (1-pi)**(n00+n10) * pi**(n01+n11)
            l_dep = (1-pi0)**n00 * pi0**n01 * (1-pi1)**n10 * pi1**n11
            lr_ind = -2 * np.log(l_ind/l_dep)
        else: lr_ind = 0
        lr_cc = lr_uc + lr_ind
        christ_p = 1 - stats.chi2.cdf(lr_cc, 2)
    except: christ_p = 0.0

    return kupiec_p, N, N_exp, christ_p

def encontrar_orden_arma(serie, max_p=3, max_q=3):
    """Busca los órdenes p y q óptimos para ARMA(p, q) según el criterio AIC/BIC."""
    serie_limpia = serie.dropna()
    if serie_limpia.empty: return 0, 0

    best_aic = np.inf
    best_order = (0, 0)

    for p in range(max_p + 1):
        for q in range(max_q + 1):
            if p == 0 and q == 0: continue
            try:
                model = ARIMA(serie_limpia, order=(p, 0, q), trend='c')
                if len(serie_limpia) > max(p, q):
                    results = model.fit(disp=False)
                    if results.aic < best_aic:
                        best_aic = results.aic
                        best_order = (p, q)
            except:
                continue

    try:
        model_const = ARIMA(serie_limpia, order=(0, 0, 0), trend='c')
        results_const = model_const.fit(disp=False)
        aic_constante = results_const.aic
        if aic_constante < best_aic:
            return 0, 0
        else:
            return best_order
    except:
        return best_order


# 3. GENERACIÓN DE GRÁFICOS (IZQUIERDA)

def generar_graficos_izquierda(nombre, serie, alpha=0.05):
    p_opt, q_opt = encontrar_orden_arma(serie)
    mean_model = 'ARMA' if p_opt > 0 or q_opt > 0 else 'Constant'
    print(f"-> {nombre}: ARMA óptimo: ({p_opt}, {q_opt}) -> Mean: {mean_model}")

    adf_p, lb_pre, arch_pre = get_stage1_stats(serie)

    dist_map = {'Normal': 'normal', 't-Student': 't', 'GED': 'ged'}
    results = {}
    serie_sc = serie * 100

    for name, dist in dist_map.items():
        try:
            # --- CORRECCIÓN DE ERROR AQUÍ: PASAR ARGUMENTOS CONDICIONALMENTE ---
            arch_args = {
                'vol': 'Garch', 'p': 1, 'q': 1,
                'dist': dist,
                'mean': mean_model,
            }

            if mean_model == 'ARMA':
                arch_args['lags'] = p_opt
                arch_args['m_out'] = q_opt

            am = arch_model(serie_sc, **arch_args)
            # -------------------------------------------------------------------

            res = am.fit(disp='off')

            params = res.params
            aic = res.aic
            resid_std = res.std_resid
            arch_post_p = het_arch(resid_std)[1]
            lb_post_p = acorr_ljungbox(resid_std.dropna(), lags=[10], return_df=True)['lb_pvalue'].iloc[0]
            stab = params.get('alpha[1]',0) + params.get('beta[1]',0)

            cond_vol = res.conditional_volatility
            mu = params['mu']
            if dist=='normal': q = am.distribution.ppf(alpha)
            else: q = am.distribution.ppf(alpha, params.get('nu', 100))
            var_sc = -(mu + cond_vol * q)
            var_series = var_sc / 100

            kp, N, Ne, cp = get_stage3_stats(var_series, serie, alpha)
            ratio_fallo = N / len(serie) if len(serie) > 0 else 0

            results[name] = {
                'aic': aic, 'var': var_series, 'params': params,
                'stab': stab, 'arch_post': arch_post_p, 'lb_post': lb_post_p,
                'kupiec': kp, 'N': N, 'Ne': Ne, 'christ': cp,
                'ratio': ratio_fallo, 'p_opt': p_opt, 'q_opt': q_opt
            }
        except Exception as e:
            print(f"!!!Error al ajustar ARMA({p_opt},{q_opt})-GARCH(1,1) con {name} para {nombre}: {e}")
            pass

    if not results:
        print(f"!!! ERROR FATAL: No se pudo ajustar ningún modelo GARCH para {nombre}.")
        return None, None, None

    best = min(results, key=lambda k: results[k]['aic'])
    d_best = results[best]

    # --- FIGURA ---
    fig = plt.figure(figsize=(8, 11))
    gs = gridspec.GridSpec(3, 1, height_ratios=[0.25, 0.50, 0.25], hspace=0.35)

    # 1. ACF / PACF de los retornos ORIGINALES
    gs_sub = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0], wspace=0.25)
    ax_acf = fig.add_subplot(gs_sub[0])
    plot_acf(serie, ax=ax_acf, title="ACF (Retornos)", zero=False, auto_ylims=True, lags=15)
    ax_pacf = fig.add_subplot(gs_sub[1])
    plot_pacf(serie, ax=ax_pacf, title="PACF (Retornos)", zero=False, auto_ylims=True, lags=15)
    for ax in [ax_acf, ax_pacf]:
        ax.tick_params(labelsize=8); ax.grid(True, alpha=0.3)
        ax.set_title(ax.get_title(), fontsize=10, fontweight='bold')

    # 2. VaR
    ax_var = fig.add_subplot(gs[1])
    ax_var.axhline(0, color='black', lw=1)
    ax_var.plot(serie.index, serie, color=COLORS['returns'], label='Retornos', lw=0.5, alpha=0.5)
    for n, d in results.items():
        ax_var.plot(serie.index, -d['var'], color=COLORS[n], label=n, lw=1.5)
    ax_var.set_title(f"VaR 95% - {nombre} (ARMA({p_opt},{q_opt})-GARCH(1,1))", fontsize=11, fontweight='bold')
    ax_var.legend(loc='lower left', ncol=3, fontsize=8, frameon=True)
    ax_var.set_xlim(serie.index[0], serie.index[-1])
    ax_var.grid(True, alpha=0.4)

    # 3. TABLA PARÁMETROS
    ax_tab = fig.add_subplot(gs[2])
    ax_tab.axis('off')

    ar_names = [f'ar[{i}]' for i in range(1, p_opt + 1)]
    ma_names = [f'ma[{i}]' for i in range(1, q_opt + 1)]
    mean_params_cols = ar_names + ma_names
    mean_cols_names = [f'AR({i})' for i in range(1, p_opt + 1)] + [f'MA({i})' for i in range(1, q_opt + 1)]
    if not mean_cols_names: mean_cols_names = ['Media']
    if not mean_params_cols: mean_params_cols = ['mu']

    cols = ['Model', 'AIC', 'Alpha', 'Beta'] + mean_cols_names + (['Shape (nu)'] if any(d['params'].get('nu') for d in results.values()) else [])
    rows_txt = []
    for n, d in results.items():
        p = d['params']
        row = [n, f"{d['aic']:.0f}", f"{p.get('alpha[1]',0):.3f}", f"{p.get('beta[1]',0):.3f}"]

        for param_name in mean_params_cols:
            row.append(f"{p.get(param_name, p.get('mu', np.nan)):.3f}")

        if 'Shape (nu)' in cols:
            row.append(f"{p.get('nu', np.nan):.2f}")

        rows_txt.append(row)

    tbl = ax_tab.table(cellText=rows_txt, colLabels=cols, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.5)
    ax_tab.set_title(f"Parámetros ARMA({p_opt},{q_opt})-GARCH(1,1) Estimados", fontsize=10, pad=5, fontweight='bold')

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)

    stats_info = {
        'stage1': (adf_p, lb_pre, arch_pre),
        'best_name': best,
        'data': d_best,
        'all_models': results,
        'p_opt': p_opt, 'q_opt': q_opt
    }

    return buf, stats_info, d_best

# 4. DATOS Y PDF

def obtener_datos():
    print(">>> Descargando Datos Diarios...")
    try:
        df = yf.download(TICKERS, start=FECHA_INICIO, end=FECHA_FIN, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            try: df = df.xs('Adj Close', level=0, axis=1)
            except: df = df.xs('Close', level=0, axis=1)
        else:
            try: df = df['Adj Close']
            except: df = df['Close']
        df = df.dropna(how="all")
    except Exception as e:
        print(f"!!! Error al descargar datos de Yahoo Finance: {e}")
        return pd.DataFrame(), pd.DataFrame()

    print(">>> Resampleando a datos semanales (cierre de viernes)...")
    precios_sem = df.resample('W-FRI').last().dropna(how='all')
    precios_sem = precios_sem.ffill().dropna()

    if precios_sem.empty:
        print("!!! Error: DataFrame de precios semanales vacío.")
        return pd.DataFrame(), pd.DataFrame()

    norm = precios_sem / precios_sem.iloc[0]
    if len(PESOS) != len(TICKERS):
        raise ValueError("La lista de pesos no coincide con la lista de tickers.")

    weighted_cols = [col for col in TICKERS if col in precios_sem.columns]
    current_weights = np.array([PESOS[TICKERS.index(col)] for col in weighted_cols])

    port = (norm[weighted_cols] * current_weights).sum(axis=1)
    precios_sem["PORTAFOLIO"] = port * precios_sem[weighted_cols].iloc[0].mean()
    cols = weighted_cols + ["PORTAFOLIO"]
    precios = precios_sem[cols]

    retornos = np.log(precios / precios.shift(1)).dropna()

    return precios, retornos

def descargar_fund(tickers):
    d = {}
    for t in tickers:
        try:
            if t=="PORTAFOLIO": d[t]={"Price":"-","PE":"-","Target":"-","Upgrade":"-"}; continue
            i = yf.Ticker(t).info
            prev = i.get('previousClose')
            tgt = i.get('targetMedianPrice')
            up = np.log(tgt/prev)*100 if (prev and tgt) else np.nan
            d[t] = {"Price": f"${prev}", "PE": f"{i.get('trailingPE',0):.1f}",
                    "Target": f"${tgt}", "Upgrade": f"{up:.1f}%" if not np.isnan(up) else "-"}
        except: d[t]={"Price":"-","PE":"-","Target":"-","Upgrade":"-"}
    return d

def calcular_metricas(ret, bench):
    if ret.empty: return {}
    mean = ret.mean()*52; std = ret.std()*np.sqrt(52)
    sharpe = (mean-0.037)/std if std!=0 else 0
    var95 = np.percentile(ret, 5)*100; var99 = np.percentile(ret, 1)*100
    beta = 1.0
    if bench is not None:
        c = ret.index.intersection(bench.index)
        if len(c)>10: beta = np.cov(ret.loc[c], bench.loc[c])[0,1] / np.var(bench.loc[c])
    return {"Beta": beta, "Volatilidad": std, "Sharpe": sharpe, "VaR 95": var95, "VaR 99": var99}


# 5. GENERACIÓN PDF

COLOR_BARRA = colors.HexColor("#0B3D91")
COLOR_LINEA = colors.HexColor("#808080")
NOMBRE_PDF = "Informe_Final_Centrado_Completo_SEMANAL_ARMA_GARCH.pdf"

def crear_marco_pagina(pdf, i, total_paginas):
    ancho, alto = A4
    pdf.setFillColor(COLOR_BARRA); pdf.rect(0, alto-3*cm, ancho, 3*cm, fill=1, stroke=0)
    pdf.setFillColor(colors.white)
    pdf.setFont("Helvetica-Bold", 16); pdf.drawString(1.5*cm, alto-1.2*cm, "Análisis cuantitativo y desempeño de activos")
    pdf.setFont("Helvetica", 10); pdf.drawString(1.5*cm, alto-2.0*cm, f"Fecha: {datetime.now().strftime('%Y-%m-%d')} | Frecuencia: SEMANAL")
    pdf.setFont("Helvetica-Bold", 12); pdf.drawRightString(ancho-1.5*cm, alto-1.2*cm, "Grupo 1 - Inversiones")
    pdf.setFont("Helvetica", 10); pdf.drawRightString(ancho-1.5*cm, alto-2.0*cm, "Autor: Carlos Caballero")

    pdf.setFillColor(colors.black)
    pdf.setFont("Helvetica", 8)
    pdf.drawString(1.5*cm, 1*cm, "Fuente: Yahoo Finance | Documento Académico")
    pdf.drawRightString(ancho-1.5*cm, 1*cm, f"Página {i} de {total_paginas}")

def generar_pdf():
    precios, retornos = obtener_datos()

    if retornos.empty:
        print("No se generó el PDF porque no hay datos de retornos disponibles.")
        return

    fund = descargar_fund(precios.columns)
    bench = retornos["SPY"] if "SPY" in retornos.columns else None

    pdf = canvas.Canvas(NOMBRE_PDF, pagesize=A4)
    ancho, alto = A4

    activos_a_analizar = precios.columns
    total_paginas = len(activos_a_analizar)

    for i, col in enumerate(activos_a_analizar, start=1):
        crear_marco_pagina(pdf, i, total_paginas)

        L_MARGIN = 1.5 * cm
        R_MARGIN = 1.5 * cm
        TOP_Y = alto - 3.5 * cm
        BOTTOM_Y = 2.5 * cm
        CONTENT_WIDTH = ancho - L_MARGIN - R_MARGIN
        WL = CONTENT_WIDTH * 0.68
        WR = CONTENT_WIDTH - WL - 0.5*cm
        XL = L_MARGIN
        XR = XL + WL + 0.5*cm

        pdf.setStrokeColor(COLOR_LINEA); pdf.setLineWidth(0.5)
        pdf.line(XR-0.25*cm, BOTTOM_Y, XR-0.25*cm, TOP_Y)

        Y = TOP_Y
        pdf.setFillColor(colors.black); pdf.setFont("Helvetica-Bold", 14)
        pdf.drawString(XL, Y, col); Y -= 0.6*cm

        est_desc = ParagraphStyle("D", fontName="Helvetica", fontSize=10, leading=12)
        pd = Paragraph(DESCRIPCIONES.get(col, "-"), est_desc)
        pd.wrap(WL, 2*cm); pd.drawOn(pdf, XL, Y-pd.height)
        Y -= (pd.height + 0.3*cm)

        asset_ret = retornos[col].dropna()
        if not asset_ret.empty:
            img_buf, info, best_data = generar_graficos_izquierda(col, asset_ret)

            if img_buf:
                p_opt, q_opt = info['p_opt'], info['q_opt']
                est_bull = ParagraphStyle("B", fontName="Helvetica", fontSize=9, leading=11, leftIndent=10)

                # Revisar si se eliminó la autocorrelación: Ljung-Box post-modelo > 0.05
                lb_status = "✅ Eliminada" if best_data['lb_post'] > 0.05 else "⚠️ No eliminada"

                bullets = [
                    f"&bull; <b>Modelo Óptimo:</b> ARMA({p_opt},{q_opt})-GARCH(1,1) con {info['best_name']}",
                    f"&bull; <b>AC en Residuos:</b> {lb_status}",
                    f"&bull; <b>Violaciones VaR:</b> {int(best_data['N'])}/{int(best_data['Ne'])}",
                    f"&bull; <b>Tasa de Fallo:</b> {best_data['ratio']:.2%} (Obj: 5%)"
                ]
                for b in bullets:
                    pb = Paragraph(b, est_bull)
                    pb.wrap(WL, 1*cm); pb.drawOn(pdf, XL, Y-pb.height)
                    Y -= (pb.height + 0.1*cm)

                Y -= 0.2*cm

                img = ImageReader(img_buf)
                h_img = Y - BOTTOM_Y
                pdf.drawImage(img, XL, Y-h_img, width=WL, height=h_img, preserveAspectRatio=True, anchor='n')

                YR = TOP_Y

                est_r = ParagraphStyle("R", fontName="Helvetica", fontSize=9, alignment=1)
                pr = Paragraph(f"<b>Serie:</b> {col}<br/><b>Rango:</b> {FECHA_INICIO}<br/>a {FECHA_FIN}", est_r)
                pr.wrap(WR, 2*cm); pr.drawOn(pdf, XR, YR-pr.height)
                YR -= (pr.height + 0.5*cm)

                est_tab = TableStyle([
                    ("BACKGROUND",(0,0),(-1,0),COLOR_BARRA), ("TEXTCOLOR",(0,0),(-1,0),colors.white),
                    ("FONT",(0,0),(-1,-1),"Helvetica",8), ("ALIGN",(0,0),(-1,-1),"CENTER"),
                    ("BOX",(0,0),(-1,-1),0.5,colors.black), ("INNERGRID",(0,0),(-1,-1),0.25,colors.grey)
                ])

                met = calcular_metricas(asset_ret, bench)
                dm = [["Métricas (Anualizadas)", ""]]
                for k,v in met.items(): dm.append([k, f"{v:.2f}" + ("%" if "VaR" in k else "")])
                tm = Table(dm, colWidths=[WR*0.55, WR*0.45]); tm.setStyle(est_tab)
                tm.wrapOn(pdf, WR, alto); tm.drawOn(pdf, XR, YR-tm._height)
                YR -= (tm._height + 0.5*cm)

                fd = fund.get(col, {})
                dfun = [["Fundam.", ""]]
                for k in ["Price","PE","Target","Upgrade"]: dfun.append([k, str(fd.get(k,"-"))])
                tf = Table(dfun, colWidths=[WR*0.55, WR*0.45]); tf.setStyle(est_tab)
                tf.wrapOn(pdf, WR, alto); tf.drawOn(pdf, XR, YR-tf._height)
                YR -= (tf._height + 0.5*cm)

                pnot = Paragraph(f"<b>Nota:</b> {NOTAS.get(col, '-')}", est_r)
                pnot.wrap(WR, 3*cm); pnot.drawOn(pdf, XR, YR-pnot.height)
                YR -= (pnot.height + 0.5*cm)

                adf, lb_pre, arch_pre = info['stage1']
                best = info['best_name']
                d = info['data']

                txt_pruebas = f"""
                <b>ETAPA 1: Diagnóstico (Retornos)</b><br/>
                &bull; ADF (Estacionario): p={adf:.4f}<br/>
                &bull; Ljung-Box (AC Pre-Mod): p={lb_pre:.4f}<br/>
                &bull; ARCH-LM (H.Cond): p={arch_pre:.4f}<br/>
                <br/>
                <b>ETAPA 2: Ajuste ARMA({p_opt},{q_opt})-GARCH(1,1)</b><br/>
                &bull; Distribución: <b>{best}</b><br/>
                &bull; Suma $\\alpha + \\beta$: {d['stab']:.3f} (&lt;1)<br/>
                &bull; AIC: {d['aic']:.0f}<br/>
                &bull; Ljung-Box (Resid. Std.): p={d['lb_post']:.3f} <br/>
                &bull; ARCH-LM (Resid. Std.): p={d['arch_post']:.3f}<br/>
                <br/>
                <b>ETAPA 3: Backtesting (VaR 95%)</b><br/>
                &bull; Kupiec (No Cond.): p={d['kupiec']:.3f}<br/>
                &bull; Christoffersen (Cond.): p={d['christ']:.3f}
                """

                est_pruebas = ParagraphStyle("P", fontName="Helvetica", fontSize=8, leading=9)
                pp = Paragraph(txt_pruebas, est_pruebas)
                pp.wrap(WR, 10*cm)

                if YR - pp.height > BOTTOM_Y:
                    pp.drawOn(pdf, XR, YR-pp.height)
                else:
                    pp.drawOn(pdf, XR, BOTTOM_Y + 0.2*cm)
            else:
                 pdf.setFont("Helvetica", 10)
                 pdf.drawString(XL, Y - 1.0*cm, f"!!! Error: Falló el ajuste del modelo ARMA-GARCH para {col}.")
        else:
            pdf.setFont("Helvetica", 10)
            pdf.drawString(XL, Y, "Sin Datos de Retornos.")

        pdf.showPage()

    pdf.save()
    print(f"\u2705 PDF FINAL GENERADO: {NOMBRE_PDF}")

if __name__ == "__main__":
    generar_pdf()
