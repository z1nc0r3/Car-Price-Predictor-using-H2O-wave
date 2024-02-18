from h2o_wave import main, app, Q, ui, on, run_on
import h2o
from h2o.automl import H2OAutoML

# h2o.init()


@app("/predictor")
async def serve(q: Q):
    # First time a browser comes to the app
    if not q.client.initialized:
        await init(q)
        q.client.initialized = True

    # Other browser interactions
    await run_on(q)
    await q.page.save()


async def init(q: Q) -> None:
    q.client.cards = set()
    q.client.dark_mode = False

    q.page["meta"] = ui.meta_card(
        box="",
        title="My Wave App",
        theme="light",
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
                                "horizontal",
                                direction=ui.ZoneDirection.ROW,
                                justify="center",
                                wrap="between",
                            ),
                            ui.zone(
                                "vertical",
                                size="1",
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

    await home(q)


@on()
async def home(q: Q):
    clear_cards(q)

    add_card(
        q,
        "make",
        ui.form_card(
            box="top_horizontal",
            items=[
                ui.dropdown(
                    required=True,
                    name="make",
                    label="Make",
                    choices=[
                        ui.choice(name=x, label=x) for x in ["Chevrolet", "Ford", "Honda", "Nissan", "Toyota"]
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
                    label="Model",
                    choices=[
                        ui.choice(name=x, label=x) for x in ["Altima", "Civic", "Camry", "F-150", "Silverado"]
                    ],
                ),
            ],
        ),
    )

    add_card(
        q,
        "year",
        ui.form_card(
            box="horizontal",
            items=[
                ui.textbox(
                    required=True,
                    name="year",
                    label="Year",
                    value="",
                    placeholder="Year",
                )
            ],
        ),
    )

    add_card(
        q,
        "mileage",
        ui.form_card(
            box="horizontal",
            items=[
                ui.textbox(
                    required=True,
                    name="mileage",
                    label="Mileage",
                    value="",
                    placeholder="Mileage",
                )
            ],
        ),
    )

    add_card(
        q,
        "condition",
        ui.form_card(
            box="horizontal",
            items=[
                ui.choice_group(
                    required=True,
                    name="condition",
                    label="Condition",
                    choices=[
                        ui.choice(name="fair", label="Fair"),
                        ui.choice(name="good", label="Good"),
                        ui.choice(name="excellent", label="Excellent"),
                    ],
                )
            ],
        ),
    )


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


def add_search_box(q: Q, msg):
    q.page["search_box"] = ui.form_card(
        box="2 2 10 2",
        items=[
            ui.textbox(
                name="search_box_input",
                label="Book Name",
                value=q.args.search_box_input,
            ),
            ui.buttons(
                items=[
                    ui.button(
                        name="search",
                        label="Search",
                        primary=True,
                        icon="BookAnswers",
                    ),
                    ui.button(name="find_books", label="Find Book", primary=False),
                ]
            ),
            ui.text(msg, size="m", name="msg_text"),
        ],
    )


def header(q: Q):
    q.page["header"] = ui.header_card(
        box="header",
        title="Car Price Prediction System",
        subtitle="Predict the price of a car using given features",
        icon="Car",
        items=[
            ui.link(
                name="github_btn",
                path="https://github.com/ChathurindaRanasinghe/book-recommendation-system_using_h2o-wave.git",
                label="GitHub",
                button=True,
            )
        ],
    )


def footer(q: Q):
    caption = """Made with 💛 by Lasith Manujitha using H2O Wave"""
    q.page["footer"] = ui.footer_card(
        box="footer",
        caption=caption,
    )
