"""
Analyse senior e-commerce - Dashboard enrichi
- Google Trends (12 mois + prévision simple)
- 4 sources TrendsMCP
- Scraping Amazon (Apify)
- Radar, prix, tendances
- Interprétations intégrées au dashboard

Auteur : Tahiry
"""

import requests
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from dotenv import load_dotenv
from openai import OpenAI
from google.colab import userdata
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

# ============================================================
# 1. INSTALLATION
# ============================================================
!pip install pytrends scipy openai scikit-learn -q

# ============================================================
# 2. CHARGEMENT DES CLÉS
# ============================================================
try:
    GROQ_API_KEY = userdata.get('GROQ_API_KEY')
    TRENDSMCP_API_KEY = userdata.get('TRENDSMCP_API_KEY')
    APIFY_API_KEY = userdata.get('APIFY_API_KEY')
    print("✅ Clés chargées")
except:
    load_dotenv()
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')
    TRENDSMCP_API_KEY = os.getenv('TRENDSMCP_API_KEY')
    APIFY_API_KEY = os.getenv('APIFY_API_KEY')

KEYWORD = "laptop"

# ============================================================
# 3. GOOGLE TRENDS + PRÉVISION
# ============================================================
from pytrends.request import TrendReq

pytrends = TrendReq(hl='en-US', tz=360)
pytrends.build_payload(kw_list=[KEYWORD], timeframe='today 12-m')
data_google = pytrends.interest_over_time()
if 'isPartial' in data_google.columns:
    data_google = data_google.drop(columns=['isPartial'])
df_google = data_google[[KEYWORD]].reset_index()
df_google.columns = ['date', 'interest']
df_google['date_num'] = (df_google['date'] - df_google['date'].min()).dt.days

# Prévision simple (3 mois)
model = LinearRegression()
model.fit(df_google[['date_num']], df_google['interest'])
future_days = (df_google['date_num'].max() + 90) - df_google['date_num'].min()
forecast_value = model.predict([[df_google['date_num'].max() + future_days]])[0]
forecast_value = max(0, min(100, forecast_value))

google_mean = df_google['interest'].mean()
google_trend_last = df_google['interest'].iloc[-1]
google_trend_first = df_google['interest'].iloc[0]
google_variation = ((google_trend_last - google_trend_first) / google_trend_first) * 100
print(f"📈 Google Trends: moyenne {google_mean:.1f} | var {google_variation:+.1f}% | prévision 3mois {forecast_value:.0f}")

# ============================================================
# 4. TRENDSMCP
# ============================================================
url = "https://api.trendsmcp.ai/api"
headers = {"Authorization": f"Bearer {TRENDSMCP_API_KEY}", "Content-Type": "application/json"}

def get_trend(source):
    try:
        data = {"source": source, "keyword": KEYWORD}
        response = requests.post(url, json=data, headers=headers, timeout=10)
        body = response.json().get("body")
        if not body or body == "null":
            return None
        body_list = json.loads(body)
        return body_list[-1].get("value") if body_list else None
    except:
        return None

sources = ["google shopping", "youtube", "google search", "google news"]
trend_values = {s: get_trend(s) for s in sources}
trend_values = {k: v for k, v in trend_values.items() if v is not None}
amazon_value = get_trend("amazon")

top_source = max(trend_values, key=trend_values.get) if trend_values else None
top_score = trend_values.get(top_source, 0) if top_source else 0
avg_trend = sum(trend_values.values()) / len(trend_values) if trend_values else 0
print(f"📊 Tendances: meilleure source = {top_source} ({top_score:.0f}) | moyenne = {avg_trend:.1f}")

# ============================================================
# 5. APIFY
# ============================================================
def scrape_amazon_products():
    actor_id = "delicious_zebu~amazon-product-data-scraper"
    api_url = f"https://api.apify.com/v2/actors/{actor_id}/runs?token={APIFY_API_KEY}"
    run_input = {"Keywords": [KEYWORD], "maxResults": 1, "minRating": 4.5}
    try:
        response = requests.post(api_url, json=run_input)
        if response.status_code != 201:
            return []
        run_id = response.json()["data"]["id"]
        for _ in range(30):
            status = requests.get(f"https://api.apify.com/v2/actor-runs/{run_id}", params={"token": APIFY_API_KEY}).json()
            if status["data"]["status"] == "SUCCEEDED":
                break
            time.sleep(2)
        return requests.get(f"https://api.apify.com/v2/actor-runs/{run_id}/dataset/items", params={"token": APIFY_API_KEY, "format": "json"}).json()
    except Exception as e:
        print(f"⚠️ Apify: {e}")
        return []

amazon_products = scrape_amazon_products()
prices = []
for p in amazon_products[:8]:
    try:
        price_str = p.get('price', '0').replace('$', '').split()[0]
        prices.append(float(price_str))
    except:
        pass
avg_price = sum(prices) / len(prices) if prices else 0
price_min = min(prices) if prices else 0
price_max = max(prices) if prices else 0
print(f"💰 Produits: n={len(amazon_products)} | prix moyen=${avg_price:.0f}")

# ============================================================
# 6. DASHBOARD ENRICHI
# ============================================================
fig = plt.figure(figsize=(18, 12))
fig.suptitle(f"📊 DASHBOARD DÉCISIONNEL - '{KEYWORD.upper()}'", fontsize=16, fontweight='bold')

# 1. Tendances avec mini interprétation
ax1 = fig.add_subplot(2, 2, 1)
bars = ax1.bar(trend_values.keys(), trend_values.values(), color=['#34a853', '#ff0000', '#4285f4', '#fbbc04'])
ax1.set_title('📈 Tendances actuelles par plateforme', fontsize=12)
ax1.set_ylabel('Intérêt (0-100)')
ax1.set_ylim(0, 100)
for bar, val in zip(bars, trend_values.values()):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f"{val:.0f}", ha='center')
interpret1 = f"✅ Canal dominant : {top_source.replace('google ', '').title()} ({top_score:.0f})\n📉 Canal à surveiller : {min(trend_values, key=trend_values.get).replace('google ', '').title()}"
ax1.text(0.5, -0.2, interpret1, transform=ax1.transAxes, fontsize=9, ha='center', va='top', color='#2c3e50', bbox=dict(boxstyle="round,pad=0.3", facecolor='#f0f0f0'))

# 2. Google Trends + prévision
ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(df_google['date'], df_google['interest'], color='#4285f4', linewidth=2, label='Historique')
last_date = df_google['date'].iloc[-1]
forecast_date = last_date + timedelta(days=90)
ax2.plot([last_date, forecast_date], [df_google['interest'].iloc[-1], forecast_value], 'r--', linewidth=2, label='Prévision 3 mois')
ax2.set_title('📉 Google Trends (12 mois + prévision)', fontsize=12)
ax2.set_xlabel('Date')
ax2.set_ylabel('Intérêt (0-100)')
ax2.legend()
ax2.grid(True, alpha=0.3)
interpret2 = f"📈 Tendance : {google_variation:+.1f}% sur 12 mois\n🔮 Prévision 3 mois : {forecast_value:.0f} ({'hausse' if forecast_value > google_trend_last else 'baisse'})"
ax2.text(0.5, -0.2, interpret2, transform=ax2.transAxes, fontsize=9, ha='center', va='top', color='#2c3e50', bbox=dict(boxstyle="round,pad=0.3", facecolor='#f0f0f0'))

# 3. Prix des produits
ax3 = fig.add_subplot(2, 2, 3)
if prices:
    titles = [p.get('title', '')[:25] for p in amazon_products[:len(prices)]]
    bars = ax3.barh(titles, prices, color='#ff9900')
    ax3.set_title(f'💰 Prix des produits Amazon (moyen = ${avg_price:.0f})', fontsize=12)
    ax3.set_xlabel('Prix (USD)')
    ax3.axvline(x=avg_price, color='red', linestyle='--', label=f'Moyenne ${avg_price:.0f}')
    ax3.legend()
    seuil_budget = 300
    produits_budget = sum(1 for p in prices if p < seuil_budget)
    interpret3 = f"🎯 {produits_budget}/{len(prices)} produits < ${seuil_budget}\n💎 Prix max : ${price_max:.0f} | min : ${price_min:.0f}"
    ax3.text(0.5, -0.2, interpret3, transform=ax3.transAxes, fontsize=9, ha='center', va='top', color='#2c3e50', bbox=dict(boxstyle="round,pad=0.3", facecolor='#f0f0f0'))
else:
    ax3.text(0.5, 0.5, 'Aucun produit', ha='center', va='center')

# 4. Radar (si Amazon absent, on l'affiche aussi)
ax4 = fig.add_subplot(2, 2, 4, projection='polar')
labels = [s.replace('google ', '').title() for s in trend_values.keys()]
values = list(trend_values.values())
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
values_radar = values + [values[0]]
angles_radar = angles + [angles[0]]
ax4.plot(angles_radar, values_radar, 'o-', linewidth=2, color='#4285f4')
ax4.fill(angles_radar, values_radar, alpha=0.25, color='#4285f4')
ax4.set_xticks(angles)
ax4.set_xticklabels(labels, size=9)
ax4.set_ylim(0, 100)
ax4.set_title('🎯 Radar des performances', size=12, pad=20)
min_src = min(trend_values, key=trend_values.get)
max_src = max(trend_values, key=trend_values.get)
interpret4 = f"✅ Point fort : {max_src.replace('google ', '').title()}\n⚠️ Point faible : {min_src.replace('google ', '').title()}"
ax4.text(0.5, -0.2, interpret4, transform=ax4.transAxes, fontsize=9, ha='center', va='top', color='#2c3e50', bbox=dict(boxstyle="round,pad=0.3", facecolor='#f0f0f0'))

plt.tight_layout()
plt.savefig("dashboard.png", dpi=150)
plt.close()
print("📊 Dashboard enrichi généré avec interprétations intégrées")

# ============================================================
# 7. ANALYSE IA AVANCÉE
# ============================================================
trends_text = "\n".join([f"- {k.replace('google ', '').title()}: {v}/100" for k, v in trend_values.items()])
products_text = "\n".join([f"- {p.get('title', '')[:45]}: {p.get('price', '')}" for p in amazon_products[:4]])

prompt = f"""
**ANALYSE STRATÉGIQUE E-COMMERCE** pour "{KEYWORD}"

**MÉTRIQUES CLÉS:**
- Google Trends (moyenne 12 mois): {google_mean:.1f}
- Variation Google Trends (12 mois): {google_variation:+.1f}%
- Prévision 3 mois: {forecast_value:.0f}
- Plateforme dominante: {top_source} ({top_score:.0f}/100)
- Prix moyen produits: ${avg_price:.0f}
- Fourchette de prix: ${price_min:.0f} - ${price_max:.0f}

**DÉTAIL DES TENDANCES:**
{trends_text}

**TOP PRODUITS:**
{products_text}

**OBJECTIF:** Produire un rapport stratégique pour un e-commerçant vendant des {KEYWORD}.

**RÉPONSE EN FRANÇAIS AVEC:**
1. DIAGNOSTIC STRATÉGIQUE (3-4 phrases sur la santé du marché)
2. 3 RECOMMANDATIONS ACTIONNABLES (chiffrées, priorisées)
3. PRÉDICTION (évolution 6 mois)
4. RISQUES à surveiller
"""

client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": prompt[:4000]}],
    temperature=0.3
)
analysis = response.choices[0].message.content
print("✅ Analyse IA générée")

# ============================================================
# 8. README
# ============================================================
readme = f"""# 📊 Analyse Décisionnelle E-commerce - '{KEYWORD}'

**Date :** {datetime.now().strftime('%Y-%m-%d %H:%M')}

## 🔑 MÉTRIQUES CLÉS

| Métrique | Valeur | Interprétation |
|----------|--------|----------------|
| Google Trends (moyenne 12 mois) | {google_mean:.1f} | Intérêt soutenu |
| Variation 12 mois | {google_variation:+.1f}% | Marché en {"hausse" if google_variation > 0 else "baisse"} |
| Prévision 3 mois | {forecast_value:.0f} | Poursuite de la tendance |
| Plateforme la plus active | {top_source} ({top_score:.0f}) | Canal prioritaire |
| Prix moyen | ${avg_price:.0f} | Positionnement |
| Fourchette de prix | ${price_min:.0f} - ${price_max:.0f} | Segmentation |

## 📈 DASHBOARD INTERACTIF

![Dashboard](dashboard.png)

## 🎯 RECOMMANDATIONS IA

{analysis}

---
*Sources : Google Trends, TrendsMCP (4 plateformes), Apify (Amazon).*
"""

with open("README.md", "w", encoding="utf-8") as f:
    f.write(readme)

print("\n✅ Terminé! dashboard.png, README.md")
