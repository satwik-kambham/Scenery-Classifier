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
    if (data.epoch == -1 && data.cost == -1) {
        eventSource.close()
        document.getElementById('save').disabled = false

    } else if (data != "Depleted") {
        addData(myChart, data.epoch, data.cost)
    }
}

document.getElementById('file-selector').addEventListener('change', (e) => {
    url = URL.createObjectURL(e.target.files[0])
    document.getElementById('image-preview').src = url
})

document.getElementById('upload-image').addEventListener('click', (e) => {
    document.getElementById('test').disabled = false
})

document.getElementById('test').addEventListener('click', () => {
    fetch('/predict')
    .then(response => response.json())
    .then(data => {
        document.getElementById('pred').innerText = data.label
        document.getElementById('conf').innerText = data.prediction
    })
})

document.getElementById('save').addEventListener('click', () => {
    fetch('/store')
})
