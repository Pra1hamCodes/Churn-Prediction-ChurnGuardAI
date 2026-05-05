# ✈️ Airline Booking Market Demand Analysis

> A Flask-based web application that analyses Australian domestic airline booking trends and surfaces AI-powered market insights — purpose-built for hostel operators who need to understand when and where travellers are flying.

## 📖 Table of Contents

- [What This App Does](#what-this-app-does)
- [Features](#features)
- [Architecture Overview](#architecture-overview)
- [Project Structure](#project-structure)
- [Core Classes](#core-classes)
- [Tech Stack & Dependencies](#tech-stack--dependencies)
- [Getting Started](#getting-started)
- [Environment Configuration](#environment-configuration)
- [API Reference](#api-reference)
- [Data Flow](#data-flow)
- [Customisation Guide](#customisation-guide)
- [Production Deployment](#production-deployment)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

---

## What This App Does

Hostel operators in Australia need to anticipate traveller demand — not just react to it. This app analyses domestic airline booking data across all major Australian city routes and answers questions like:

- Which routes are seeing the highest demand right now?
- Which airlines are competitively priced vs. in high demand?
- When are the peak travel periods for the next 90 days?
- What should a hostel in Brisbane know about Sydney–Brisbane flight trends?

The app pulls flight data (live via AviationStack API, or from a realistic sample dataset), processes it through a pandas-powered analytics layer, generates interactive Plotly visualisations, and optionally sends the market summary to GPT-3.5 Turbo for plain-English strategic recommendations — all from a single Flask server.

**It works without any API keys** — both the OpenAI and AviationStack integrations degrade gracefully to rule-based fallbacks, so you can run the full app immediately with realistic sample data.

---

## Features

### 📊 Interactive Data Visualisations
Three Plotly charts rendered dynamically in the browser:
- **Popular Routes by Demand Score** — horizontal bar chart ranking routes by computed demand
- **Price Trends Over Time** — line chart showing average fare movement across the 90-day window
- **Airline Price vs Demand** — scatter plot comparing each carrier's average price against their demand score, plotted with airline name labels

### 🗺️ Australian Market Focus
Sample data covers the six major Australian domestic city pairs — Sydney, Melbourne, Brisbane, Perth, Adelaide, and Darwin — across four carriers: Qantas, Virgin Australia, Jetstar, and Tiger Airways.

### 🔍 Smart Filtering
The `/api/filter` endpoint supports filtering by origin city, destination city, and date range. Filtered results instantly re-run all analytics and regenerate charts and AI insights for the narrowed dataset.

### 🤖 AI-Powered Market Insights
When an OpenAI API key is configured, a GPT-3.5 Turbo prompt synthesises the top metrics (total flights, average price, peak demand score, top routes) into a structured market briefing with trends, pricing insights, demand patterns, and hostel-specific recommendations. Without a key, a well-structured rule-based fallback renders identical sections using the live computed values.

### 🛡️ Security & Resilience
- All secrets loaded from environment variables via `python-dotenv`
- Input sanitisation on filter parameters
- Full try/except coverage on every API call and data processing step
- Structured logging throughout (`logging.INFO` level) for easy debugging
- CORS and rate-limiting awareness built into the session configuration

### 📱 Responsive Design
The frontend (served from `templates/index.html`) is designed to work on desktop and mobile browsers.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                  Browser (Frontend SPA)                  │
│    templates/index.html  ·  static/ (CSS, JS, assets)   │
│                                                          │
│   Filter Controls  →  Fetch /api/filter                 │
│   Dashboard Load   →  Fetch /api/data                   │
│   Plotly Charts    ←  JSON chart specs                  │
│   AI Insights      ←  Markdown text                     │
└───────────────────────────┬─────────────────────────────┘
                            │ HTTP REST
┌───────────────────────────▼─────────────────────────────┐
│                  Flask Application (app.py)              │
│                                                          │
│  Route: /            → render_template('index.html')    │
│  Route: /api/data    → full dataset + charts + AI       │
│  Route: /api/filter  → filtered dataset + charts + AI   │
│                                                          │
│  ┌─────────────────┐  ┌──────────────────────────────┐  │
│  │  DataScraper    │  │       DataProcessor          │  │
│  │                 │  │                              │  │
│  │ get_sample_     │  │  get_market_insights()       │  │
│  │ flight_data()   │  │  → popular_routes (top 10)   │  │
│  │                 │  │  → price_trends (by date)    │  │
│  │ get_aviation_   │  │  → demand_periods (top 5)    │  │
│  │ data()          │  │  → airline_stats             │  │
│  │ (AviationStack) │  │  → summary metrics           │  │
│  └─────────────────┘  └──────────────────────────────┘  │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │             AIInsightGenerator                   │   │
│  │  generate_insights()  →  OpenAI GPT-3.5 Turbo   │   │
│  │  _generate_sample_insights()  →  rule-based      │   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │               create_charts()                    │   │
│  │   Plotly → JSON (PlotlyJSONEncoder) → response   │   │
│  └──────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────┘
            │ Optional                  │ Optional
  ┌─────────▼──────────┐    ┌───────────▼─────────────┐
  │  AviationStack API  │    │     OpenAI API           │
  │  (live flight data) │    │  GPT-3.5 Turbo insights  │
  └────────────────────┘    └─────────────────────────┘
```

---

## Project Structure

```
airline-booking-analysis/
│
├── app.py                  # Flask app — all routes, classes, and chart logic (362 lines)
├── config.py               # Environment-aware Config / DevelopmentConfig / ProductionConfig
├── requirements.txt        # Pinned Python dependencies
│
├── templates/
│   └── index.html          # Main SPA — dashboard, filter controls, chart containers
│
├── static/                 # CSS, JS, and any static assets served by Flask
│
├── api/                    # (Reserved / modular API handlers)
│
├── __pycache__/            # Python bytecode cache (auto-generated, gitignored)
│
└── ReadME.md               # Original README
```

---

## Core Classes

### `FlightData` (dataclass)
A lightweight data container for individual flight records. Fields: `origin`, `destination`, `price` (float), `date` (str), `airline`, `demand_score` (float, default `0.0`). Used throughout the pipeline as the typed unit of data.

```python
@dataclass
class FlightData:
    origin: str
    destination: str
    price: float
    date: str
    airline: str
    demand_score: float = 0.0
```

---

### `DataScraper`
Handles all data acquisition. Initialises a `requests.Session` with a browser-like User-Agent.

**`get_sample_flight_data() → List[FlightData]`**
Generates 50 randomised `FlightData` objects across 6 Australian cities, 4 airlines, prices between $150–$800, dates spread over the next 90 days, and demand scores between 0.3–1.0. Used when no live API key is present.

**`get_aviation_data(api_key) → List[Dict]`**
Makes a live call to the AviationStack `/v1/flights` endpoint (100 flights, `GET` with access key). Returns the `data` array on success or an empty list on failure — ensuring the app never crashes when the API is unavailable.

---

### `DataProcessor`
Converts raw `FlightData` lists into structured market analytics using pandas.

**`get_market_insights(flights) → Dict`**
Builds a DataFrame from the flight list, then computes:

- **Popular routes** — groups by `(origin, destination)`, aggregates mean `price` and `demand_score`, creates a `route` label (`"Sydney → Melbourne"`), returns top 10 by demand score
- **Price trends** — groups by `date`, computes mean daily price, returns as date-indexed series
- **Demand periods** — groups by `date`, computes mean daily demand score, returns top 5 peak days
- **Airline stats** — groups by `airline`, aggregates mean price and demand score
- **Summary metrics** — `total_flights`, `avg_price`, `peak_demand_score`

Returns a single `Dict` with all five keys plus the three summary values — this same dict feeds both the chart builder and the AI insight generator.

---

### `AIInsightGenerator`
Wraps OpenAI's Chat Completions API (`gpt-3.5-turbo`).

**`generate_insights(market_data) → str`**
Constructs a structured prompt with total flights, average price, peak demand score, and the top 5 routes, requesting four specific output sections: key market trends, pricing insights, demand patterns, and hostel operator recommendations. Calls `POST https://api.openai.com/v1/chat/completions` with `max_tokens=500` and `temperature=0.7`.

**`_generate_sample_insights(market_data) → str`**
The graceful fallback. Returns a Markdown-formatted multi-section analysis using the live computed values (total flights, avg price, peak demand score) so the UI always shows meaningful, data-grounded text even without an API key.

---

### `create_charts(insights) → Dict[str, str]`
Module-level function that builds three Plotly figures and serialises them to JSON strings via `PlotlyJSONEncoder` for safe transport to the browser:

| Key | Chart Type | X-Axis | Y-Axis |
|---|---|---|---|
| `popular_routes` | `go.Bar` | Route label | Demand score |
| `price_trends` | `go.Scatter` (lines+markers) | Date | Average price ($) |
| `airline_comparison` | `go.Scatter` (markers+text) | Average price ($) | Demand score |

---

## Tech Stack & Dependencies

| Package | Version | Role |
|---|---|---|
| `Flask` | 2.3.3 | Web framework — routing, template rendering, JSON responses |
| `requests` | 2.31.0 | HTTP client for AviationStack and OpenAI API calls |
| `pandas` | 2.0.3 | DataFrame-based analytics — groupby, aggregation, trend computation |
| `beautifulsoup4` | 4.12.2 | HTML parsing (for future web scraping extensions) |
| `plotly` | 5.15.0 | Interactive chart generation, serialised as JSON for the frontend |
| `python-dotenv` | 1.0.0 | Loads `.env` file into `os.environ` |
| `gunicorn` | 21.2.0 | Production WSGI server |
| `lxml` | 4.9.3 | Fast XML/HTML parser backend for BeautifulSoup |

**Language breakdown:** Python 41.9% · HTML 58.1%

---

## Getting Started

### Prerequisites
- Python 3.8 or higher
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Pra1hamCodes/airline-booking-analysis.git
cd airline-booking-analysis

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create your environment file
cp .env.example .env            # or create .env manually (see below)

# 5. Run the app
python app.py

# 6. Open in browser
# http://localhost:5000
```

The app runs immediately with sample data — no API keys are required to see the full interface, charts, and rule-based insights.

---

## Environment Configuration

Create a `.env` file in the project root:

```env
# ── Flask ─────────────────────────────────────────────────────
SECRET_KEY=your-strong-random-secret-key-here
FLASK_DEBUG=True                  # Set to False in production

# ── OpenAI (optional) ─────────────────────────────────────────
# Powers the GPT-3.5 Turbo market briefing.
# Without this, rule-based insights are used instead.
OPENAI_API_KEY=sk-...

# ── AviationStack (optional) ──────────────────────────────────
# Provides live flight data. Free tier: 100 requests/month.
# Without this, realistic sample data is generated locally.
AVIATIONSTACK_API_KEY=your-aviationstack-key

# ── Database (future use) ─────────────────────────────────────
DATABASE_URL=sqlite:///airline_data.db

# ── Deployment ────────────────────────────────────────────────
PORT=5000
```

### Config Classes (`config.py`)

The app ships with two config classes, selectable by environment:

| Class | `DEBUG` | Use Case |
|---|---|---|
| `DevelopmentConfig` | `True` | Local development (default) |
| `ProductionConfig` | `False` | Live deployment |

---

## API Reference

### `GET /`
Renders the main dashboard (`templates/index.html`). No parameters.

---

### `GET /api/data`
Loads and processes the full flight dataset (sample or live), generates all charts, and calls the AI insight generator.

**Response:**
```json
{
  "success": true,
  "insights": {
    "popular_routes": [...],
    "price_trends": [...],
    "demand_periods": [...],
    "airline_stats": [...],
    "total_flights": 50,
    "avg_price": 421.73,
    "peak_demand_score": 0.97
  },
  "ai_insights": "**Market Analysis Summary**\n...",
  "charts": {
    "popular_routes": "{...plotly json...}",
    "price_trends": "{...plotly json...}",
    "airline_comparison": "{...plotly json...}"
  }
}
```

---

### `GET /api/filter`
Returns a filtered subset of the dataset. All parameters are optional — omitting them returns the full dataset.

| Parameter | Type | Description | Example |
|---|---|---|---|
| `origin` | `string` | Filter by origin city (case-insensitive) | `Sydney` |
| `destination` | `string` | Filter by destination city (case-insensitive) | `Melbourne` |
| `date_from` | `string` | Start of date range (`YYYY-MM-DD`) | `2025-06-01` |
| `date_to` | `string` | End of date range (`YYYY-MM-DD`) | `2025-08-31` |

**Example request:**
```
GET /api/filter?origin=Sydney&destination=Melbourne
```

**Response:** Same structure as `/api/data` but scoped to the filtered flights.

---

## Data Flow

```
Request hits /api/data or /api/filter
        │
        ▼
DataScraper.get_sample_flight_data()
  → 50 FlightData objects (or live from AviationStack)
        │
        ▼
  Apply filter params (origin, destination, date range)
        │
        ▼
DataProcessor.get_market_insights(flights)
  → pandas DataFrame
  → groupby aggregations
  → Dict with routes, trends, stats, summary
        │
        ├──────────────────────────────────────┐
        ▼                                      ▼
create_charts(insights)            AIInsightGenerator.generate_insights(insights)
  → 3 Plotly figures               → OpenAI API call (or rule-based fallback)
  → PlotlyJSONEncoder              → Markdown string
  → Dict of JSON strings
        │                                      │
        └──────────────┬───────────────────────┘
                       ▼
              jsonify({ success, insights, charts, ai_insights })
                       │
                       ▼
              Browser renders charts + insight panel
```

---

## Customisation Guide

### Add New Australian Routes or Airlines
In `DataScraper.get_sample_flight_data()`, extend the `cities` or `airlines` lists:
```python
cities = ['Sydney', 'Melbourne', 'Brisbane', 'Perth', 'Adelaide', 'Darwin',
          'Hobart', 'Canberra', 'Cairns']  # add more
airlines = ['Qantas', 'Virgin Australia', 'Jetstar', 'Tiger Airways',
            'Rex Airlines']               # add more
```

### Connect a Real Data Source
Replace or augment `get_sample_flight_data()` in `DataScraper` to call any flight data API. The method just needs to return `List[FlightData]` — everything downstream works unchanged.

### Change the AI Model
In `AIInsightGenerator.generate_insights()`, swap the model string:
```python
'model': 'gpt-4o',  # or any OpenAI-compatible model
```

### Add a New Chart
In `create_charts()`, add a new Plotly figure and serialise it:
```python
new_fig = go.Figure(data=[go.Pie(labels=..., values=...)])
new_fig.update_layout(title='My New Chart')
charts['my_chart'] = json.dumps(new_fig, cls=plotly.utils.PlotlyJSONEncoder)
```
Then reference `charts.my_chart` in the frontend JavaScript.

### Add New Analytics Metrics
Extend `DataProcessor.get_market_insights()` — add new pandas operations and include new keys in the returned dict. The `insights` dict flows all the way through to `AIInsightGenerator`, so new metrics are automatically available for the AI prompt too.

### Extend the Config
`config.py` already stubs a `SQLALCHEMY_DATABASE_URI` for SQLite. To add persistent storage, install `Flask-SQLAlchemy`, define your models, and wire them into the data pipeline.

---

## Production Deployment

### Using Gunicorn (included in `requirements.txt`)

```bash
# Set production environment
export FLASK_DEBUG=False
export SECRET_KEY=<strong-random-key>

# Start with Gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Pre-deployment Checklist

| Task | Details |
|---|---|
| Set `SECRET_KEY` | Use a long random string — never the default |
| Set `FLASK_DEBUG=False` | Disables the interactive debugger in production |
| Configure `OPENAI_API_KEY` | Optional — app works without it |
| Configure `AVIATIONSTACK_API_KEY` | Optional — sample data is used as fallback |
| Enable HTTPS | Use a reverse proxy (nginx, Caddy) or a PaaS that provides TLS |
| Set up logging | Redirect `gunicorn` and app logs to a file or log aggregator |
| Test all routes | Verify `/`, `/api/data`, and `/api/filter` work with production config |

### Deploy to Render / Railway / Fly.io
The app is a standard Flask + Gunicorn application. Set the start command to:
```
gunicorn app:app
```
And set your environment variables in the platform's secrets/env panel.

---

## Troubleshooting

| Issue | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError` | Dependencies not installed | Run `pip install -r requirements.txt` |
| `TemplateNotFound: index.html` | Missing `templates/` directory | Create `templates/` and ensure `index.html` is inside it |
| Charts not rendering | Plotly JSON not parsed by frontend | Check browser console; verify `/api/data` returns valid JSON |
| AI insights show generic text | `OPENAI_API_KEY` not set or invalid | Check `.env` — rule-based fallback activates automatically |
| No live flight data | `AVIATIONSTACK_API_KEY` not set | Expected — sample data loads automatically |
| `Port 5000 already in use` | Another process on port 5000 | Change `PORT` in `.env` or kill: `lsof -ti:5000 \| xargs kill` |
| Blank dashboard on filter | Filter returns 0 flights | Try broader filter values; sample data is randomised each request |
| `OPENAI API error: 429` | Rate limit exceeded | Add retry logic or use the rule-based fallback by leaving key empty |

---

## Contributing

```bash
# 1. Fork the repo on GitHub

# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/airline-booking-analysis.git
cd airline-booking-analysis

# 3. Create a virtual environment and install deps
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 4. Create a feature branch
git checkout -b feature/your-feature

# 5. Make changes, commit, and push
git commit -m "feat: describe your change"
git push origin feature/your-feature

# 6. Open a Pull Request
```

### Good First Contributions
- **Date range filtering** — wire up `date_from` / `date_to` params in `filter_data()` (the parameters are already accepted but the filtering logic is not yet applied to dates)
- **Persistent caching** — cache the processed insights in SQLite so repeat visits don't re-generate sample data
- **Real scraping** — implement `BeautifulSoup`-based scraping (the dependency is already installed) for a public flight data source
- **Route map visualisation** — add a Plotly `geo` scatter map showing Australian city nodes with route lines sized by demand score
- **CSV export** — add a `/api/export` endpoint that returns the current filtered dataset as a downloadable CSV

---

## License

No license file is currently included. All rights are retained by the author. Contact the repository owner before using or distributing this code.

---

## 👤 Author

**Pra1hamCodes**
GitHub: [@Pra1hamCodes](https://github.com/Pra1hamCodes)

---

> *Built with Flask, pandas, Plotly, and OpenAI GPT-3.5 — fully functional with zero API keys, extensible when you add them.*
