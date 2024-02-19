from h2o_wave import main, app, Q, ui, on, run_on
import h2o
from h2o.automl import H2OAutoML
import time, numpy as np
from collections import defaultdict
import pandas as pd

h2o.init()


@app("/predictor")
async def serve(q: Q):
    # First time a browser comes to the app
    if not q.client.initialized:
        await init(q)
        q.client.initialized = True

    await home(q)

    if q.args.predict:
        await predict_button_click(q)

    if q.args.submit and q.args.file_upload:
        local_path = await upload_data(q)
        handle_table(q, local_path)

    # Other browser interactions
    await run_on(q)
    await q.page.save()


async def init(q: Q) -> None:
    q.client.cards = set()
    q.client.dark_mode = False

    q.page["meta"] = ui.meta_card(
        box="",
        title="My Wave App",
        themes=[
            ui.theme(
                name="my-awesome-theme",
                primary="#ffdd51",
                text="#e8e1e1",
                card="#373737",
                page="#070b1a",
            )
        ],
        theme="my-awesome-theme",
        layouts=[
            ui.layout(
                breakpoint="xs",
                min_height="100vh",
                max_width="1400px",
                zones=[
                    ui.zone("header"),
                    ui.zone(
                        "content",
                        size="1",
                        zones=[
                            ui.zone(
                                "model_importer",
                                direction=ui.ZoneDirection.COLUMN,
                                size="1",
                            ),
                            ui.zone(
                                "top_horizontal",
                                direction=ui.ZoneDirection.ROW,
                                justify="center",
                                wrap="between",
                                align="center",
                            ),
                            ui.zone(
                                "middle_horizontal",
                                direction=ui.ZoneDirection.ROW,
                                justify="center",
                                wrap="between",
                            ),
                            ui.zone(
                                "bottom_horizontal",
                                direction=ui.ZoneDirection.ROW,
                                justify="center",
                                size="1",
                                wrap="between",
                                zones=[
                                    ui.zone(
                                        "bottom_horizontal_left",
                                        size="130px",
                                        justify="center",
                                    ),
                                    ui.zone(
                                        "bottom_horizontal_right",
                                        size="500px",
                                        justify="center",
                                    ),
                                ],
                            ),
                        ],
                    ),
                    ui.zone(name="footer"),
                ],
            )
        ],
    )

    header(q)
    footer(q)


@on()
async def home(q: Q) -> None:
    if "file_upload" in q.args:
        add_card(
            q,
            "model_importer",
            ui.form_card(
                box="model_importer",
                items=[
                    ui.text(f"file_upload={q.args.file_upload}"),
                    ui.button(name="show_upload", label="Back", primary=True),
                ],
            ),
        )
    else:
        add_card(
            q,
            "model_importer",
            ui.form_card(
                box="model_importer",
                items=[
                    ui.file_upload(
                        required=True,
                        name="file_upload",
                        label="Select one or more files to upload",
                        compact=True,
                        multiple=False,
                        file_extensions=["csv"],
                        max_file_size=1,
                        max_size=15,
                    ),
                    ui.button(name="submit", label="Submit", primary=True),
                ],
            ),
        )

    add_card(
        q,
        "make",
        ui.form_card(
            box="top_horizontal",
            items=[
                ui.dropdown(
                    required=True,
                    name="make",
                    placeholder="Select Make",
                    label="Make",
                    value=q.args.make,
                    choices=[
                        ui.choice(name=x, label=x)
                        for x in ["Chevrolet", "Ford", "Honda", "Nissan", "Toyota"]
                    ],
                )
            ],
        ),
    )

    add_card(
        q,
        "model",
        ui.form_card(
            box="top_horizontal",
            items=[
                ui.dropdown(
                    required=True,
                    name="model",
                    placeholder="Select Model",
                    label="Model",
                    value=q.args.model,
                    choices=[
                        ui.choice(name=x, label=x)
                        for x in ["Altima", "Civic", "Camry", "F-150", "Silverado"]
                    ],
                ),
            ],
        ),
    )

    add_card(
        q,
        "year",
        ui.form_card(
            box="middle_horizontal",
            items=[
                ui.textbox(
                    required=True,
                    name="year",
                    label="Enter Build Year",
                    value=q.args.year,
                    placeholder="Year",
                )
            ],
        ),
    )

    add_card(
        q,
        "mileage",
        ui.form_card(
            box="middle_horizontal",
            items=[
                ui.textbox(
                    required=True,
                    name="mileage",
                    label="Enter Mileage",
                    value=q.args.mileage,
                    placeholder="Mileage",
                )
            ],
        ),
    )

    add_card(
        q,
        "condition",
        ui.form_card(
            box="middle_horizontal",
            items=[
                ui.choice_group(
                    required=True,
                    name="condition",
                    label="Condition",
                    value=q.args.condition,
                    choices=[
                        ui.choice(name="fair", label="Fair"),
                        ui.choice(name="good", label="Good"),
                        ui.choice(name="excellent", label="Excellent"),
                    ],
                )
            ],
        ),
    )

    add_card(
        q,
        "submit",
        ui.form_card(
            box="bottom_horizontal_left",
            items=[
                ui.buttons(
                    items=[
                        ui.button(name="predict", label="Predict", primary=True),
                    ]
                )
            ],
        ),
    )

    add_card(
        q,
        "predicted_price_card",
        ui.form_card(
            box="bottom_horizontal_right",
            items=[
                ui.text(
                    content="Please fill all the fields to predict the price of a car.",
                    name="predicted_price",
                    size="l",
                ),
            ],
        ),
    )


async def upload_data(q: Q):
    uploaded_files_dict = defaultdict()

    uploaded_file_path = q.args.file_upload
    filename = uploaded_file_path[0].split("/")[-1]
    uploaded_files_dict[filename] = uploaded_file_path[0]

    try:
        local_path = await q.site.download(uploaded_file_path[0], "./Data")
        print(f"File downloaded to: {local_path}")
    except Exception as e:
        print(f"Error downloading file: {e}")

    return local_path


def make_markdown_row(values):
    return f"| {' | '.join([str(x) for x in values])} |"


def make_markdown_table(fields, rows):
    return "\n".join(
        [
            make_markdown_row(fields),
            make_markdown_row("-" * len(fields)),
            "\n".join([make_markdown_row(row) for row in rows]),
        ]
    )


def handle_table(q, local_path):
    if "file_upload" in q.args:
        df = pd.read_csv(local_path)
        add_card(
            q,
            "table",
            ui.form_card(
                box="model_importer",
                items=[
                    ui.text(
                        make_markdown_table(
                            fields=df.columns.tolist(),
                            rows=list(map(str, df.values.tolist()[i]) for i in df.index[0:100]),
                        )
                    ),
                ],
            ),
        )


async def predict_button_click(q: Q):
    try:
        make = q.args.make
        model = q.args.model
        year = int(q.args.year)
        mileage = int(q.args.mileage)
        condition = q.args.condition
        print(make, model, year, mileage, condition)

        await update_predicted_price(q, make, model, year, mileage, condition)

    except Exception as e:
        q.page["meta"].dialog = ui.dialog(
            title="Error!",
            name="error_dialog",
            items=[
                ui.text(f"Please fill all the fields with valid data! {str(e)}"),
            ],
            closable=True,
        )

    return


async def update_predicted_price(
    q: Q, make: str, model: str, year: int, mileage: int, condition: str
):
    predicted_price = await predict_price(make, model, year, mileage, condition)

    add_card(
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
    make: str, model: str, year: int, mileage: int, condition: str
) -> float:
    model = h2o.import_mojo(
        "./Model/StackedEnsemble_AllModels_1_AutoML_1_20240218_204552.zip"
    )
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
    data_frame = h2o.H2OFrame(
        data,
        column_names=column_names,
    )

    predictions = model.predict(data_frame)
    print(predictions)

    return round(predictions.flatten(), 2)


async def train_model(q):
    # Load the data
    df = h2o.import_file("data/car_dataset.csv")

    # Split the data into training and testing sets
    train, test = df.split_frame(ratios=[0.8])

    # Identify the response and predictor variables
    y = "Price"
    x = df.columns
    x.remove(y)

    # Run AutoML
    aml = H2OAutoML(max_runtime_secs=60)
    aml.train(x=x, y=y, training_frame=train)

    # Get the best model
    best_model = aml.leader

    # Make predictions
    predictions = best_model.predict(test)
    print(predictions)

    path = "./Model/model.zip"
    best_model.save_mojo(path)

    # Return the predictions
    return predictions


def add_card(q, name, card) -> None:
    q.client.cards.add(name)
    q.page[name] = card


def clear_cards(q, ignore=[]) -> None:
    for name in q.client.cards.copy():
        if name not in ignore:
            del q.page[name]
            q.client.cards.remove(name)


async def show_timer(q: Q):
    main_page = q.page["predictor"]
    max_runtime_secs = q.args.max_runtime_secs
    for i in range(1, max_runtime_secs):
        pct_complete = int(np.ceil(i / max_runtime_secs * 100))
        main_page.items = [
            ui.progress(
                label="Training Progress",
                caption=f"{pct_complete}% complete",
                value=i / max_runtime_secs,
            )
        ]
        await q.page.save()
        await q.sleep(1)


def header(q: Q):
    q.page["header"] = ui.header_card(
        box="header",
        title="Car Price Prediction System",
        subtitle="Predict the price of a car using given features",
        icon="Car",
        items=[
            ui.link(
                name="github_btn",
                path="https://github.com/z1nc0r3/Laptop-Price-Predictor-using-H2O-wave",
                label="GitHub",
                button=True,
            )
        ],
        color="primary",
    )


def footer(q: Q):
    caption = """Made with 💛 by Lasith Manujitha using H2O Wave"""
    q.page["footer"] = ui.footer_card(
        box="footer",
        caption=caption,
    )
