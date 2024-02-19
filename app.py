from h2o_wave import main, app, Q, ui, on, run_on
from collections import defaultdict
import h2o
import pandas as pd
import os
from Train.train import Trainer
from Train.predictor import Predictor

h2o.init()


@app("/predictor")
async def serve(q: Q):
    if not q.client.initialized:
        await init(q)
        q.client.initialized = True

    # Load the models from the Model directory
    q.client.aml_models = os.listdir("./Model/")
    await home(q)

    # Clear the expired cards
    q.page["meta"].dialog = None
    q.page["meta"].notification_bar = None

    trainer = Trainer()
    predictor = Predictor()

    # Event Handling
    if q.args.predict and q.args.aml_model:
        await predictor.predict_button_click(q)

    # Handle file upload and data visualization
    if q.args.submit and q.args.file_upload:
        local_path = await upload_data(q)
        q.client.local_path = local_path
        handle_table(q, local_path)

    # Train the model and save it in the Model directory
    if q.args.train_model and q.client.local_path:
        await trainer.train_model(q, q.client.local_path)

    await run_on(q)
    await q.page.save()


async def init(q: Q) -> None:
    q.client.cards = set()
    q.client.dark_mode = False

    q.page["meta"] = ui.meta_card(
        box="",
        title="Car Price Prediction System",
        themes=[
            ui.theme(
                name="my-awesome-theme",
                primary="#ffe600",
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
                            ),
                            ui.zone(
                                "top_horizontal",
                                direction=ui.ZoneDirection.ROW,
                                justify="center",
                                wrap="between",
                                align="center",
                                zones=[
                                    ui.zone(
                                        "top_horizontal_left",
                                        justify="center",
                                        size="50%",
                                    ),
                                    ui.zone(
                                        "top_horizontal_right",
                                        justify="center",
                                        size="50%",
                                    ),
                                ],
                            ),
                            ui.zone(
                                "middle_horizontal",
                                direction=ui.ZoneDirection.ROW,
                                justify="center",
                                wrap="between",
                                zones=[
                                    ui.zone(
                                        "middle_horizontal_left",
                                        size="33.33%",
                                        justify="center",
                                    ),
                                    ui.zone(
                                        "middle_horizontal_middle",
                                        size="33.33%",
                                        justify="center",
                                    ),
                                    ui.zone(
                                        "middle_horizontal_right",
                                        size="33.33%",
                                        justify="center",
                                    ),
                                ],
                            ),
                            ui.zone(
                                "model_selector",
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
    # File Upload Field
    if "file_upload" in q.args:
        add_card(
            q,
            "model_importer",
            ui.form_card(
                box="model_importer",
                items=[
                    ui.text(f"file_upload={q.args.file_upload}"),
                    ui.buttons(
                        justify="start",
                        items=[
                            ui.button(name="submit", label="Submit", primary=True),
                            ui.button(
                                name="train_model", label="Train Model", primary=True
                            ),
                        ],
                    ),
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
                        label="Select a dataset file (CSV)",
                        compact=True,
                        multiple=False,
                        file_extensions=["csv"],
                        max_file_size=5,
                        max_size=5,
                    ),
                    ui.buttons(
                        justify="start",
                        items=[
                            ui.button(name="submit", label="Submit", primary=True),
                            ui.button(
                                name="train_model", label="Train Model", primary=True
                            ),
                        ],
                    ),
                ],
            ),
        )

    # Form Fields
    add_card(
        q,
        "make",
        ui.form_card(
            box="top_horizontal_left",
            items=[
                ui.dropdown(
                    required=True,
                    name="make",
                    placeholder="Select Company",
                    label="Company",
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
            box="top_horizontal_right",
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
            box="middle_horizontal_left",
            items=[
                ui.textbox(
                    required=True,
                    name="year",
                    mask="[0-9]{4}",
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
            box="middle_horizontal_middle",
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
            box="middle_horizontal_right",
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

    # Model Selector
    add_card(
        q,
        "model_selector",
        ui.form_card(
            box="model_selector",
            items=[
                ui.dropdown(
                    required=True,
                    name="aml_model",
                    placeholder="Select Model",
                    label="AutoML Model",
                    value=q.args.aml_model,
                    choices=[ui.choice(name=x, label=x) for x in q.client.aml_models],
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

    # Prediction Card
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


""" File upload handler """


async def upload_data(q: Q):
    uploaded_files_dict = defaultdict()

    uploaded_file_path = q.args.file_upload
    filename = uploaded_file_path[0].split("/")[-1]
    uploaded_files_dict[filename] = uploaded_file_path[0]

    try:
        local_path = await q.site.download(uploaded_file_path[0], "./Data")
    except Exception as e:
        q.page["meta"].dialog = ui.dialog(
            title="Error!",
            name="error_dialog",
            items=[ui.text("File upload failed!")],
            closable=True,
        )

        return

    return local_path


""" Data Visualization """


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
        try:
            df = pd.read_csv(local_path)
        except Exception as e:
            q.page["meta"].dialog = ui.dialog(
                title="Error!",
                name="error_dialog",
                items=[
                    ui.text("Something went wrong! Please try again."),
                ],
                closable=True,
            )

            return

        add_card(
            q,
            "table",
            ui.form_card(
                box="model_importer",
                items=[
                    ui.text(
                        make_markdown_table(
                            fields=df.columns.tolist(),
                            rows=list(
                                map(str, df.values.tolist()[i]) for i in df.index[0:6]
                            ),
                        )
                    ),
                ],
            ),
        )


def add_card(q, name, card) -> None:
    q.client.cards.add(name)
    q.page[name] = card


def clear_cards(q, ignore=[]) -> None:
    for name in q.client.cards.copy():
        if name not in ignore:
            del q.page[name]
            q.client.cards.remove(name)


""" Header and Footer """


def header(q: Q):
    q.page["header"] = ui.header_card(
        box="header",
        title="Car Price Prediction System",
        subtitle="Predict the price of a car using given features",
        icon="Car",
        items=[
            ui.link(
                name="github_btn",
                path="https://github.com/z1nc0r3/Car-Price-Predictor-using-H2O-wave",
                label="GitHub",
                button=True,
            )
        ],
        color="primary",
    )


def footer(q: Q):
    caption = """ · Made with 💛 by [Lasith Manujitha](https://www.linkedin.com/in/lasith-manujitha) using H2O Wave · """
    q.page["footer"] = ui.footer_card(
        box="footer",
        caption=caption,
    )
