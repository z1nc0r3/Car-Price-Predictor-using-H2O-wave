from h2o_wave import main, app, Q, ui, on, run_on
import h2o
from h2o.automl import H2OAutoML
import time, numpy as np

# h2o.init()


@app("/predictor")
async def serve(q: Q):
    # First time a browser comes to the app
    if not q.client.initialized:
        await init(q)
        q.client.initialized = True

    """ try:
        make, model, year, mileage, condition = get_form_data()
    except Exception as e:
        # Handle the exception here
        print(f"An error occurred: {str(e)}") """

    await home(q)
    # When the user clicks the predict button
    if q.args.predict:
        await predict_button_click(q)

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
    # clear_cards(q)

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


async def predict_button_click(q: Q):
    # Extract values from form elements (or other data sources)
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
    # Implement your model prediction logic here, replacing this placeholder
    predicted_price = await predict_price(make, model, year, mileage, condition)

    # Update the predicted price card
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
    """model = h2o.import_mojo(
        "./Model/StackedEnsemble_AllModels_1_AutoML_1_20240218_204552.zip"
    )"""
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
    """ data_frame = h2o.H2OFrame(
        data,
        column_names=column_names,
    ) """

    # predictions = model.predict(data_frame)
    predictions = 12312.454
    print(predictions)

    # return round(predictions.flatten(), 2)
    return predictions


""" async def predict(q):
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

    path = './Model/model.zip'
    best_model.save_mojo(path)

    # Return the predictions
    return predictions """


@on()
async def change_theme(q: Q):
    """Change the app from light to dark mode"""
    if q.client.dark_mode:
        q.page["header"].items = [
            ui.menu(
                [ui.command(name="change_theme", icon="ClearNight", label="Dark mode")]
            )
        ]
        q.page["meta"].theme = "light"
        q.client.dark_mode = False
    else:
        q.page["header"].items = [
            ui.menu([ui.command(name="change_theme", icon="Sunny", label="Light mode")])
        ]
        q.page["meta"].theme = "h2o-dark"
        q.client.dark_mode = True


# Use for cards that should be deleted on calling `clear_cards`. Useful for routing and page updates.
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
