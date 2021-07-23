# Flask server

## Running

To run development server, run following command:

```python main.py```

Should run on http://localhost:5000/

## Adding models

Add models by putting your model file into `models` directory. It should follow this requirement:
- input: query (string)
- output: relevant document ids (list[int])

To run model, go to `main.py`, import your model and change `SELECTED_MODEL` to your model. 