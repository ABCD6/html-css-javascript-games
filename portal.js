const games = [
  { name: "Breakout Game", path: "Breakout Game/index.html" },
  { name: "Car Game", path: "Car Game/index.html" },
  { name: "Flappy Bird", path: "Flappy Bird/index.html" },
  { name: "Ping Pong", path: "Ping Pong/index.html" },
  { name: "Snake Game", path: "Snake Game/index.html" },
  { name: "Tic Tac Toe", path: "Tic Tac Toe/index.html" },
  { name: "2048", path: "2048/index.html" },
  { name: "Memory Game", path: "Memory Game/index.html" },
  { name: "Piano Tiles", path: "Piano Tiles/index.html" },
  { name: "Rock Paper Scissors", path: "Rock Paper Scissors/index.html" },
];

const grid = document.getElementById("game-grid");
games.forEach(g => {
  const card = document.createElement("div");
  card.className = "game-card";
  card.textContent = g.name;
  card.onclick = () => openGame(g.path);
  grid.appendChild(card);
});

const modal = document.getElementById("game-modal");
const frame = document.getElementById("game-frame");
const closeModal = document.getElementById("close-modal");

function openGame(path) {
  frame.src = path;
  modal.style.display = "block";
}

closeModal.onclick = () => {
  frame.src = "";
  modal.style.display = "none";
};

window.onclick = e => {
  if (e.target === modal) {
    frame.src = "";
    modal.style.display = "none";
  }
};
