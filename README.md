# Pac-Man AI with Deep Q-Learning

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-orange)
![Flask](https://img.shields.io/badge/Flask-2.0%2B-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

This project implements an AI-driven Pac-Man game using Deep Q-Learning (DQN) with PyTorch and Flask. The AI learns to navigate a 20x20 grid, eating food while avoiding ghosts. After 10 episodes, it optimizes to consistently collect food when safe. Pac-Man moves continuously until game over (collision with a ghost), delivering a smooth, nonstop gameplay experience via a web interface.

## Features
- **Continuous Movement**: Pac-Man moves nonstop using `requestAnimationFrame` for fluid animation.
- **Self-Learning AI**: Employs DQN to balance food collection and ghost avoidance, improving over episodes.
- **Optimized Behavior**: Post-10 episodes, Pac-Man prioritizes food when ghosts are >3 tiles away.
- **Web Visualization**: Built with Flask and HTML5 Canvas for real-time game rendering.

## Demo
![Pac-Man AI Gameplay](https://via.placeholder.com/400x400.png?text=Pac-Man+AI+Gameplay)  
*(Replace this with an actual GIF or screenshot of your game by uploading it to the repo and updating the URL.)*

## Prerequisites
- **Python**: 3.7 or higher
- **Dependencies**: Flask, PyTorch, NumPy
- **Browser**: Modern browser (e.g., Chrome, Firefox) with JavaScript enabled

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Coding-Shanks/pacman-ai.git
   cd pacman-ai
   ```

2. **Install Dependencies**:
   ```bash
   pip install flask torch numpy
   ```

3. **Project Structure**:
   ```
   pacman-ai/
   ├── app.py
   ├── templates/
   │   └── index.html
   ├── pacman_model.pth  (generated after training)
   └── README.md
   ```

## Usage

1. **Run the Application**:
   ```bash
   python app.py
   ```
   The Flask server starts at `http://127.0.0.1:5000`.

2. **Open in Browser**:
   - Visit `http://127.0.0.1:5000`.
   - The game begins automatically with Pac-Man moving continuously.

3. **Controls**:
   - **Reset**: Click the "Reset" button to start a new episode.
   - Game ends on ghost collision, displaying "Game Over."

4. **Observe Training**:
   - Monitor metrics (score, episode, epsilon, etc.) in real-time.
   - After 10 episodes, Pac-Man optimizes for food collection.

## How It Works
- **DQN Model**: A neural network (128-64-4 layers) predicts Q-values for actions (up, right, down, left) based on a 16-feature state vector (danger, direction, distances).
- **Training**: Explores randomly for 10 episodes (`epsilon` decays from 1.0 to 0.01), then exploits learned strategies, prioritizing food when safe.
- **Game Loop**: Frontend uses `requestAnimationFrame` to fetch steps continuously from the backend until game over.
- **Persistence**: Model saves to `pacman_model.pth` every 5 episodes.

## Code Overview

- **[`app.py`](./app.py)**: Backend logic with Flask, DQN implementation, and game mechanics.
- **[`templates/index.html`](./templates/index.html)**: Frontend with Canvas rendering and real-time metrics.

## Troubleshooting
- **Performance Lag**: Reduce grid size in `app.py` (e.g., `grid_width = 10`) or use a faster browser.
- **Model Issues**: Delete `pacman_model.pth` if corrupted and retrain.
- **Server Errors**: Check Flask logs in the terminal and ensure dependencies are installed.

## Future Enhancements
- Add manual Pac-Man controls alongside AI.
- Introduce multiple food items or power-ups.
- Visualize neural network decisions in the UI.

## Contributing
Contributions are welcome! Please:
1. Fork the repo.
2. Create a feature branch (`git checkout -b feature/YourFeature`).
3. Commit changes (`git commit -m 'Add YourFeature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a Pull Request.

## License
This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

## Acknowledgments
- Built with inspiration from classic Pac-Man and reinforcement learning tutorials.
- Thanks to the PyTorch and Flask communities for excellent tools and documentation.

---

### Notes for GitHub
1. **Replace Placeholder**: Update the demo image URL (`https://via.placeholder.com/...`) with an actual screenshot or GIF of your game. Upload it to the repo (e.g., in an `assets/` folder) and link it like `![Gameplay](./assets/gameplay.gif)`.
2. **Update Clone URL**: Replace `https://github.com/Coding-Shanks/pacman-ai.git` with your actual repository URL.
3. **Add License File**: Create a `LICENSE` file in the repo with the MIT License text if you choose that license.
4. **Badges**: The badges (Python, PyTorch, etc.) are optional but enhance the README’s visual appeal. Adjust versions as needed.

