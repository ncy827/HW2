let chart;
function initGrid(id, policy) {
    const grid = document.getElementById(id);
    grid.innerHTML = '';
    const arrows = ["↑", "↓", "←", "→"];
    for (let r = 0; r < 4; r++) {
        for (let c = 0; c < 12; c++) {
            const div = document.createElement('div');
            div.className = 'cell';
            if (r === 3 && c > 0 && c < 11) { div.className += ' cliff'; div.innerText = 'C'; }
            else if (r === 3 && c === 0) { div.className += ' start'; div.innerText = 'S'; }
            else if (r === 3 && c === 11) { div.className += ' goal'; div.innerText = 'G'; }
            else { div.innerText = arrows[policy[r][c]]; }
            grid.appendChild(div);
        }
    }
}

async function startTraining() {
    document.getElementById('status').innerText = "訓練中，請稍候...";
    const res = await fetch('/train', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({episodes: 500})
    });
    const data = await res.json();
    initGrid('q-grid', data.q_learning.policy);
    initGrid('sarsa-grid', data.sarsa.policy);

    const ctx = document.getElementById('rewardChart').getContext('2d');
    if (chart) chart.destroy();
    chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: Array.from({length: 500}, (_, i) => i),
            datasets: [
                { label: 'Q-learning', data: data.q_learning.rewards, borderColor: 'red', pointRadius: 0, fill: false },
                { label: 'SARSA', data: data.sarsa.rewards, borderColor: 'teal', pointRadius: 0, fill: false }
            ]
        },
        options: { scales: { y: { min: -100, max: 0 } } }
    });
    document.getElementById('status').innerText = "訓練完成！已儲存 cliff.jpg 與 result_sample.jpg";
}