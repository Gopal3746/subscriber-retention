# Streamlit Community Cloud deployment

1. Push this repository to GitHub.
2. Open Streamlit Community Cloud and create a new app from the repository.
3. Set the entrypoint to `dashboard/app.py`.
4. Use Python 3.12 in Advanced settings for parity with current Community Cloud defaults.
5. Deploy. The root `requirements.txt` contains all Python dependencies.
6. Add the resulting `https://<your-app>.streamlit.app` URL to the README and resume.

The dashboard reads precomputed CSV artifacts committed in `artifacts/`, so startup stays lightweight. Re-run `make build` locally whenever you update the raw data or modeling logic, then commit the refreshed artifacts.
