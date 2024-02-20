from h2o_wave import Q, ui
from h2o.automl import H2OAutoML
import h2o
import concurrent.futures
import asyncio
import os


class Trainer:
    def __init__(self):
        pass

    async def train_model(self, q: Q, local_path: str):
        try:
            df = h2o.import_file(local_path)

            # Split the dataset into training and testing sets
            train, test = df.split_frame(ratios=[0.7], seed=10)

            y = "Price"
            x = df.columns
            x.remove(y)
        except Exception as e:
            q.page["meta"].dialog = ui.dialog(
                title="Error!",
                name="error_dialog",
                items=[
                    ui.text("Something went wrong!"),
                ],
                closable=True,
            )

            return

        # Configure the AutoML model and train the model using the given dataset
        aml = H2OAutoML(max_models=10, seed=10, verbosity="info", nfolds=8)

        # Show a dialog to indicate the training process
        future = asyncio.ensure_future(self.show_progress(q))
        with concurrent.futures.ThreadPoolExecutor() as pool:
            await q.exec(pool, self.aml_train, aml, x, y, train)
        future.cancel()
        self.post_training(q)

        # Save the best trained model in the Model directory
        best_model = aml.leader

        predictions = best_model.predict(test)

        path = "./Model/"
        h2o.save_model(model=best_model, path=path, force=True)
        # best_model.save_mojo(path)
        q.client.aml_models = os.listdir(path)

        self.add_card(
            q,
            "model_selector",
            ui.form_card(
                box="model_selector",
                items=[
                    ui.dropdown(
                        required=True,
                        name="aml_model",
                        placeholder="Select Model",
                        label="Model",
                        value=q.args.aml_model,
                        choices=[
                            ui.choice(name=x, label=x) for x in q.client.aml_models
                        ],
                    )
                ],
            ),
        )

        await q.page.save()

        return predictions

    def aml_train(self, aml, x, y, train):
        aml.train(x=x, y=y, training_frame=train)

    def post_training(self, q):
        q.page["meta"].dialog = None
        q.page["meta"] = ui.meta_card(
            box="",
            notification_bar=ui.notification_bar(
                text="Model trained successfully!",
                type="success",
                position="top-right",
                name="notification_bar",
            ),
        )

    """ Show the progress of the training process"""

    async def show_progress(self, q: Q):
        for i in range(1, 25):
            q.page["meta"].dialog = ui.dialog(
                title="Training Model",
                items=[
                    ui.text(f"Training model... {i}%"),
                ],
                closable=False,
                blocking=True,
                width="300px",
            )

            await q.page.save()
            await q.sleep(1)

    def add_card(self, q, name, card) -> None:
        q.client.cards.add(name)
        q.page[name] = card
