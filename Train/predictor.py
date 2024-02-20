from h2o_wave import Q, ui
import h2o
import numpy as np


class Predictor:
    def __init__(self):
        pass

    async def predict_button_click(self, q: Q):
        q.args.predict = False
        q.page["meta"].dialog = None
        q.page["meta"].notification_bar = None

        # Predict the price of a car using the given features. User can select the model to use for the prediction.
        try:
            make = q.args.make
            model = q.args.model
            year = int(q.args.year)
            mileage = int(q.args.mileage)
            condition = q.args.condition
            aml_model = q.args.aml_model

            print(make, model, year, mileage, condition, aml_model)

            await self.update_predicted_price(
                q, make, model, year, mileage, condition, aml_model
            )

        except Exception as e:
            q.page["meta"].dialog = ui.dialog(
                title="Error!",
                name="error_dialog",
                items=[
                    ui.text(f"Please fill all the fields with valid data!"),
                ],
                closable=True,
            )

        return

    async def update_predicted_price(
        self,
        q: Q,
        make: str,
        model: str,
        year: int,
        mileage: int,
        condition: str,
        aml_model: str,
    ):
        predicted_price = await self.predict_price(
            make, model, year, mileage, condition, aml_model
        )

        # Update the predicted price card with the predicted price
        self.add_card(
            q,
            "predicted_price_card",
            ui.form_card(
                box="bottom_horizontal_right",
                items=[
                    ui.text(
                        content=f"Predicted Price: ${predicted_price}",
                        name="predicted_price",
                        size="l",
                    ),
                ],
            ),
        )

    async def predict_price(
        self,
        make: str,
        model: str,
        year: int,
        mileage: int,
        condition: str,
        aml_model: str,
    ) -> float:
        model = h2o.load_model(f"./Model/{aml_model}")
        column_names = [
            "Year",
            "Mileage",
            "Make_Chevrolet",
            "Make_Ford",
            "Make_Honda",
            "Make_Nissan",
            "Make_Toyota",
            "Model_Altima",
            "Model_Camry",
            "Model_Civic",
            "Model_F-150",
            "Model_Silverado",
            "Condition_Excellent",
            "Condition_Fair",
            "Condition_Good",
        ]

        data = [year, mileage]

        # One-hot encode the make, model and condition of the car
        for i in range(2, 7):
            if str(make) in column_names[i].lower():
                data.append(1)
            else:
                data.append(0)

        for i in range(7, 12):
            if str(model) in column_names[i].lower():
                data.append(1)
            else:
                data.append(0)

        for i in range(12, 15):
            if str(condition) in column_names[i].lower():
                data.append(1)
            else:
                data.append(0)

        data = np.array([data])
        print(data)
        data_frame = h2o.H2OFrame(
            data,
            column_names=column_names,
        )

        predictions = model.predict(data_frame)

        return round(predictions.flatten(), 2)

    def add_card(self, q, name, card) -> None:
        q.client.cards.add(name)
        q.page[name] = card
