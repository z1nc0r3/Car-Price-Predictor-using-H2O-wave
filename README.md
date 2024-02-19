# Car Price Predictor using H2O Wave

This car price prediction system, built with H2O Wave, and H2O AutoML, utilizes features like year,
mileage, make, model, and condition to estimate vehicle prices.




https://github.com/z1nc0r3/Car-Price-Predictor-using-H2O-wave/assets/64279853/b93944e0-41ef-4793-9e48-80afb84fb39f




- Train the model with custom datasets (CSV)
- Use already existing AutoML models for prediction.
- Save trained AutoML models locally.


## Running the app

### 1. Clone the repository:

``` bash
git clone https://github.com/z1nc0r3/Car-Price-Predictor-using-H2O-wave
```

### 2. Create a virtual environment:

``` bash
python3 -m venv venv
```

### 3. Activate the virtual environment:
``` bash
source venv/bin/activate
```

**windows**
``` bash
venv\Scripts\activate.bat
```
To deactivate the virtual environment use ```deactivate``` command.

### 4. Install dependencies:

``` bash
(venv) pip3 install -r requirements.txt 
```

### 5. Run the app:
``` bash
(venv) wave run app
```

### 6. View the app:
Point your favorite web browser to http://localhost:10101/predictor

