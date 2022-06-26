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

document.getElementById('load').addEventListener('click', () => {
    fetch('/load', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            fname: document.getElementById('fname').value,
        })
    })
        .then(response => response.json())
        .then(data => {
            document.getElementById('c1').innerText = data['category-1']
            document.getElementById('c2').innerText = data['category-2']
            document.getElementById('ep').innerText = data['epochs']
            document.getElementById('lr').innerText = data['learning-rate']
            document.getElementById('lgr').innerText = data['log-rate']
        })
})