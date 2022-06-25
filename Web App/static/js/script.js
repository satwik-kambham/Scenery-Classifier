const ctx = document.getElementById('epochs-v-cost');
const myChart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: [],
        datasets: [{
            label: 'Cost',
            data: [],
            backgroundColor: [
                'rgba(255, 99, 132, 0.2)',
            ],
            borderColor: [
                'rgba(255, 99, 132, 1)',
            ],
            borderWidth: 1,
            fill: true,
            cubicInterpolationMode: 'monotone',
        }]
    },
    options: {
        scales: {
            x: {
                title: {
                    display: true,
                    text: 'Epochs'
                },
            },
            y: {
                title: {
                    display: true,
                    text: 'Cost'
                }
            }
        }
    }
});

function addData(chart, label, data) {
    chart.data.labels.push(label)
    chart.data.datasets.forEach((dataset) => {
        dataset.data.push(data)
    });
    chart.update()
}

let eventSource = new EventSource('/data')
eventSource.onmessage = (e) => {
    let data = JSON.parse(e.data)
    if (data != "Depleted") {
        addData(myChart, data.epoch, data.cost)
    }
}
