# 🌊 OceanEmbed
### Satellite Embedding-Based Deep Learning Framework for Reconstruction of Subsurface Ocean Temperature

**SIH 2026 | Problem Statement: PS-26066 | Ministry of Earth Sciences (MoES) | INCOIS**

OceanEmbed is a deep learning framework designed to reconstruct depth-wise subsurface ocean temperature profiles using satellite-derived surface observations. It focuses on the North Indian Ocean and combines a dual-encoder neural network with an interactive visualization dashboard and an underwater acoustic intelligence layer.

> **Project status:** Working prototype / demo stage. The end-to-end application flow is implemented according to the project documentation. Real temperature prediction accuracy is not yet established because GPU training and scientific validation are pending.

## 📌 Problem Statement

Satellite observations provide information about ocean surface conditions, but understanding temperature variations at different underwater depths is important for oceanographic research and underwater acoustic applications.

OceanEmbed aims to estimate temperature profiles at multiple depths using surface observations and deep learning.

## 🎯 Project Objectives

- Reconstruct subsurface ocean temperature profiles from surface observations.
- Process oceanographic datasets into a standardized spatial grid.
- Predict temperatures at 15 standard depth levels.
- Estimate thermocline depth.
- Calculate sound velocity profiles and underwater acoustic characteristics.
- Visualize selected ocean locations and predicted profiles through an interactive dashboard.

## ✨ Key Features

### 1. Ocean Data Processing
- Daily satellite and oceanographic data processing.
- North Indian Ocean region cropping and spatial regridding.
- Z-score normalization and patch extraction.
- Missing sea-surface salinity value handling.
- Automated data pipeline components.

### 2. Deep Learning Model
- CNN encoder based on ResNet-50.
- Vision Transformer (ViT)-style encoder.
- Feature fusion using concatenated embeddings.
- MLP decoder for 15-depth temperature predictions.
- Auxiliary thermocline prediction head.
- Depth-weighted Huber loss.

**Documented model architecture:** Approximately 72.18 million parameters.

### 3. Underwater Acoustic Intelligence
The acoustic processing layer includes:

- Sound Velocity Profile (SVP) using the Mackenzie (1981) formula.
- Sonic Layer Depth (SLD).
- SOFAR channel axis estimation.
- Shadow-zone estimation using Snell's Law.
- Grid-level acoustic processing.

### 4. Interactive Dashboard
- Interactive North Indian Ocean map.
- Location selection using map clicks.
- Date selection for the 2022 dataset.
- Temperature profile visualization.
- Thermocline depth reference.
- React frontend integrated with a FastAPI backend.

### 5. Backend API
The FastAPI backend provides prediction and health-check endpoints with request/response validation.

## 🗺️ Geographic and Scientific Scope

| Parameter | Specification |
|---|---|
| Region | North Indian Ocean |
| Latitude | 5°N–30°N |
| Longitude | 45°E–105°E |
| Spatial resolution | 0.25° × 0.25° |
| Temporal scope | Daily observations |
| Dataset year documented | 2022 |
| Target depth levels | 15 |
| Maximum target depth | 1000 metres |

Target depths in metres:

`0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000`

## 🛰️ Datasets

The project documentation lists the following datasets:

| Dataset | Source |
|---|---|
| Sea Surface Temperature (SST) — NOAA OISST v2.1 | NOAA NCEI / Earthdata |
| Sea Surface Height / Anomaly (SSH/SLA) — DUACS | Copernicus Marine |
| Sea Surface Salinity (SSS) — SMOS + SMAP L4 | Copernicus Marine |
| Surface Currents (U, V) | GLORYS12 / Copernicus |
| Surface Winds (U, V) | Copernicus Marine |
| Training Temperature Targets — GLORYS12 | Copernicus Marine |
| ARGO Validation Profiles | Ifremer ERDDAP |
| WOA23 Climatology | NOAA NCEI |

The documented data pipeline covers the 2022 dataset. Multi-year training data and 0.25° WOA23 climatology are listed as pending enhancements.

## 🧠 Model Architecture

OceanEmbed uses a dual-encoder architecture with an MLP decoder.

1. **CNN Encoder:** Extracts spatial features from surface input patches.
2. **ViT Encoder:** Learns spatial relationships using transformer-based attention.
3. **Feature Fusion:** Concatenates CNN and ViT embeddings.
4. **MLP Decoder:** Produces temperature estimates for 15 depth levels.
5. **Thermocline Head:** Estimates thermocline depth using temperature gradients and soft-argmax.

### Training Configuration

- Framework: PyTorch 2.x
- Optimizer: AdamW
- Learning-rate scheduler: Cosine Annealing
- Gradient clipping: Maximum norm of 1.0
- Main loss: Depth-weighted Huber loss
- Auxiliary thermocline loss: Weight 0.1
- Training/validation split: 80/20

**Important:** The documentation reports a 10-epoch smoke test with training loss decreasing from 12.4 to 8.8. Full GPU training and validated model weights remain pending.

## 🛠️ Technology Stack

| Layer | Technologies |
|---|---|
| Programming languages | Python, JavaScript / JSX |
| Machine learning | PyTorch, torchvision |
| Data processing | NumPy, xarray, SciPy, netCDF4 |
| Regridding | xESMF |
| Data storage | Zarr |
| Backend | FastAPI, Pydantic, Uvicorn |
| Frontend | React 18, Vite, Tailwind CSS |
| Visualization | Recharts, Leaflet, OpenStreetMap |
| Validation | argopy / ARGO profiles |
| Notebook | Jupyter, Matplotlib |
| Version control | Git |

## 📂 Project Structure

```text
oceanembed/
├── acoustic/
│   ├── svp.py
│   ├── sld.py
│   ├── sofar.py
│   ├── shadow_zone.py
│   └── acoustic_processor.py
├── checkpoints/
├── data/
│   └── processed/
├── docs/
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Charts/
│   │   │   ├── Map/
│   │   │   └── Panels/
│   │   ├── App.jsx
│   │   └── main.jsx
│   ├── index.html
│   └── package.json
├── model/
│   ├── encoder_cnn.py
│   ├── encoder_vit.py
│   ├── decoder.py
│   ├── loss.py
│   ├── thermocline_head.py
│   └── trainer.py
├── notebooks/
│   └── demo.ipynb
├── pipeline/
│   ├── fetcher.py
│   ├── preprocessor.py
│   ├── glorys_regrid.py
│   ├── patch_extractor.py
│   ├── gap_filler.py
│   ├── validator.py
│   ├── scheduler.py
│   └── preprocess_currents_winds.py
├── serving/
│   └── api.py
└── tests/
    └── test_all.py
```

*Note: This structure reflects the documented project layout. Confirm filenames and directories against the actual repository before publishing.*

## ⚙️ Installation and Setup

### Prerequisites

- Python environment compatible with the project dependencies.
- Node.js and npm.
- Git.
- Access to the required datasets for preprocessing or training.

### 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd oceanembed
```

Replace the placeholder with your actual repository URL.

### 2. Install Backend Dependencies

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the project's dependencies:

```bash
pip install -r requirements.txt
```

*If `requirements.txt` does not exist in the repository, create it from the actual project environment before using this command.*

### 3. Start the FastAPI Backend

```bash
cd serving
uvicorn api:app --reload --port 8000
```

API documentation:

`http://127.0.0.1:8000/docs`

### 4. Start the Frontend

Open a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the local URL shown by Vite, documented as:

`http://localhost:5173`

### 5. Try the Demo

1. Start the backend and frontend.
2. Open the dashboard in your browser.
3. Select a point on the North Indian Ocean map.
4. Select a date from 2022.
5. Click **Predict Profile**.
6. View the temperature profile and thermocline reference.

**Demo limitation:** The documented demo currently uses random model weights. The resulting temperatures should not be interpreted as scientifically validated ocean observations or predictions.

## 🔌 API Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Check backend status |
| `/predict` | POST | Generate a temperature profile and thermocline estimate |
| `/depths` | GET | Return standard depth levels |
| `/docs` | GET | Explore the API using Swagger UI |

The documented `/predict` endpoint accepts a `5 × 64 × 64` surface patch and returns a 15-depth output with a thermocline estimate.

Refer to the API's Swagger documentation for the actual request schema.

## 🧪 Testing

The project documentation reports **9/9 automated tests passing** in `tests/test_all.py`.

The listed tests cover:

- Sound velocity profile shape and range.
- Sonic Layer Depth and SOFAR calculations.
- Shadow-zone count.
- CNN and ViT encoder output shapes.
- Dual-encoder model output shape.
- Non-negative Huber loss.
- Thermocline-head output shape.

These tests verify the listed computational and model-shape checks. They do not establish real-world prediction accuracy.

Run the tests from the repository root:

```bash
pytest tests/test_all.py -v
```

## 🚧 Pending Work and Future Improvements

The following tasks are identified in the project documentation:

- Publish the GitHub repository and complete the README.
- Implement an uncertainty prediction head.
- Add JWT authentication and role-based access.
- Integrate PostgreSQL and Redis.
- Containerize the services using Docker Compose.
- Upgrade the map using Mapbox and Deck.gl.
- Implement Science and Tactical dashboard modes.
- Evaluate predictions against ARGO observations using RMSE, MAE, Bias and R².
- Perform full GPU training on GLORYS 2022 data.
- Expand the training dataset to 2010–2022.
- Obtain WOA23 climatology at 0.25° resolution.

## 🌍 Potential Applications

Subject to successful model training and validation, the framework may support:

- Subsurface ocean-temperature analysis.
- Oceanographic research and visualization.
- Thermocline and sound-velocity profile analysis.
- Research into underwater acoustic propagation.

These are intended applications, not claims of operational or scientifically validated performance.

## 🎓 Project Information

- **Project:** OceanEmbed
- **Problem Statement:** PS-26066
- **Organization:** Ministry of Earth Sciences (MoES)
- **Department:** Indian National Centre for Ocean Information Services (INCOIS)
- **SIH:** Smart India Hackathon 2026
- **Category:** Software
- **Theme:** Space Technology
- **Team:** Ocean Valley

## 🤝 Contributing

Contributions, suggestions and improvements are welcome. Please open an issue to discuss significant changes before submitting a pull request.

## 📄 License

A license should be added to the repository before redistributing the project. The project documentation does not establish which license is intended; add the correct license file and update this section accordingly.

---

**OceanEmbed — Reconstructing the Ocean Beneath the Surface.**
