# Neural Chess Interface

A web-based interface for a neural network chess bot. This application allows you to play against or analyze positions using a custom-trained PyTorch model.

## Prerequisites

* Python 3.8 or higher
* A trained model checkpoint file (e.g., `.pt` file) placed in the `checkpoints/` folder.

## Installation

1.  **Clone or download** this repository to your local machine.
2.  **Install the required dependencies** using pip:

    ```bash
    pip install -r requirements.txt
    ```

## How to Run

1.  **Start the Flask server**:
    Run the following command in your terminal:

    ```bash
    python app.py
    ```

2.  **Open the Interface**:
    Once the server starts, open your web browser and navigate to:
    [http://127.0.0.1:5000](http://127.0.0.1:5000)

## Usage

1.  **Load a Model**:
    * On the left sidebar, look for the **Settings** card.
    * Select your model file from the dropdown menu (these are loaded from the `checkpoints/` directory).
    * Click **Load**.
    * *Note: If you have multiple versions, the CPU-trained checkpoints often provide more accurate results for this architecture.*

2.  **Game Modes**:
    * **Analysis**: Setup any board position. The AI will predict the top moves for the current side to play.
    * **Play Bot**: Play a game against the AI. You can choose to play as White or Black.

3.  **Adjust Settings**:
    * **Elo**: Adjust the target rating to see how a 1000 Elo player might move versus a 2000 Elo player.
    * **Top-k**: Change how many candidate moves the AI displays.

## Project Structure

* `app.py`: The web server and main entry point.
* `checkpoints/`: Place your trained model `.pt` files here.
* `inference.py`: Logic for loading models and generating predictions.
* `model_and_dataset.py`: Model architecture (`SimpleChessNet`) and board encoding logic.
* `requirements.txt`: List of Python dependencies.