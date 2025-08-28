document.getElementById('youtubeForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const formData = new FormData(this);
    const resultDiv = document.getElementById('result');
    
    try {
        resultDiv.innerHTML = 'Processing...';
        
        const response = await fetch('https://jb-llamaindex.onrender.com/process-youtube', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        resultDiv.innerHTML = `<pre>${JSON.stringify(result, null, 2)}</pre>`;
    } catch (error) {
        resultDiv.innerHTML = `Error: ${error.message}`;
    }
});