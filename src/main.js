import './style.css';

const form = document.getElementById('analyze-form');
const textarea = document.getElementById('news-text');
const charCount = document.getElementById('char-count');
    
    textarea.addEventListener('input', () => {
        charCount.textContent = textarea.value.length.toLocaleString();
    });

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        const text = textarea.value.trim();
        if (!text) return;
        
        const modelSelect = document.getElementById('model-select');
        const selectedModel = modelSelect.value;
        const selectedModelName = modelSelect.options[modelSelect.selectedIndex].text;

        const submitBtn = document.getElementById('submit-btn');
        const loading = document.getElementById('loading');
        const resultContainer = document.getElementById('result-container');
        const errorContainer = document.getElementById('error-message');
        const mockWarning = document.getElementById('mock-warning');
        
        const resultBox = document.getElementById('result-box');
        const resultGlow = document.getElementById('result-glow');
        const resultIconWrapper = document.getElementById('result-icon-wrapper');
        const resultIcon = document.getElementById('result-icon');
        const resultLabel = document.getElementById('result-label');
        const confidenceText = document.getElementById('confidence-text');
        const confidenceBar = document.getElementById('confidence-bar');
        
        const probRealText = document.getElementById('prob-real-text');
        const probFakeText = document.getElementById('prob-fake-text');
        const inferenceTimeText = document.getElementById('inference-time-text');
        const uncertaintyWarning = document.getElementById('uncertainty-warning');
        const modelUsedText = document.getElementById('model-used-text');

        submitBtn.disabled = true;
        submitBtn.classList.add('opacity-50', 'cursor-not-allowed');
        loading.classList.remove('hidden');
        resultContainer.classList.add('hidden');
        errorContainer.classList.add('hidden');
        mockWarning.classList.add('hidden');
        if(uncertaintyWarning) uncertaintyWarning.classList.add('hidden');
        
        resultIconWrapper.classList.remove('scale-100');
        resultIconWrapper.classList.add('scale-0');
        confidenceBar.style.width = '0%';

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: text, model: selectedModel })
            });

            const data = await response.json();
            
            if (data.error) throw new Error(data.error);
            if (data.mock) mockWarning.classList.remove('hidden');

            const isReal = data.prediction.toLowerCase().includes('real');
            
            resultLabel.textContent = data.prediction;
            confidenceText.textContent = data.confidence.toFixed(1) + '%';
            if(probRealText) probRealText.textContent = (data.prob_real !== undefined ? data.prob_real : 0).toFixed(1) + '%';
            if(probFakeText) probFakeText.textContent = (data.prob_fake !== undefined ? data.prob_fake : 0).toFixed(1) + '%';
            if(inferenceTimeText) inferenceTimeText.textContent = (data.inference_time_sec !== undefined ? (data.inference_time_sec * 1000).toFixed(3) : '0.000') + ' ms';
            if(modelUsedText) modelUsedText.textContent = selectedModelName;

            if (uncertaintyWarning) {
                if (data.confidence < 60) {
                    uncertaintyWarning.classList.remove('hidden');
                } else {
                    uncertaintyWarning.classList.add('hidden');
                }
            }
            
            if (isReal) {
                resultBox.className = 'flex flex-col items-center justify-center p-8 rounded-3xl border border-emerald-200 bg-white relative overflow-hidden transition-all duration-500 shadow-[0_10px_40px_rgba(16,185,129,0.1)]';
                resultGlow.className = 'absolute inset-0 opacity-10 transition-colors duration-1000 bg-emerald-50';
                
                resultIconWrapper.className = 'w-20 h-20 rounded-full flex items-center justify-center mb-6 shadow-md transform transition-transform duration-700 bg-emerald-100 scale-0';
                resultIcon.className = 'fa-solid fa-check text-4xl text-emerald-600';
                
                resultLabel.className = 'text-4xl font-extrabold mb-8 tracking-tight text-emerald-600';
                confidenceBar.className = 'h-full rounded-full transition-all duration-1000 ease-out relative overflow-hidden bg-gradient-to-r from-emerald-500 to-teal-400';
                confidenceText.className = 'text-2xl font-bold font-mono text-emerald-700';
            } else {
                resultBox.className = 'flex flex-col items-center justify-center p-8 rounded-3xl border border-rose-200 bg-white relative overflow-hidden transition-all duration-500 shadow-[0_10px_40px_rgba(244,63,94,0.1)]';
                resultGlow.className = 'absolute inset-0 opacity-10 transition-colors duration-1000 bg-rose-50';
                
                resultIconWrapper.className = 'w-20 h-20 rounded-full flex items-center justify-center mb-6 shadow-md transform transition-transform duration-700 bg-rose-100 scale-0';
                resultIcon.className = 'fa-solid fa-xmark text-4xl text-rose-600';
                
                resultLabel.className = 'text-4xl font-extrabold mb-8 tracking-tight text-rose-600';
                confidenceBar.className = 'h-full rounded-full transition-all duration-1000 ease-out relative overflow-hidden bg-gradient-to-r from-rose-500 to-red-500';
                confidenceText.className = 'text-2xl font-bold font-mono text-rose-700';
            }

            resultContainer.classList.remove('hidden');

            setTimeout(() => {
                resultIconWrapper.classList.remove('scale-0');
                resultIconWrapper.classList.add('scale-100');
                confidenceBar.style.width = data.confidence + '%';
            }, 50);

        } catch (error) {
            document.getElementById('error-text').textContent = error.message || 'An error occurred during analysis.';
            errorContainer.classList.remove('hidden');
        } finally {
            submitBtn.disabled = false;
            submitBtn.classList.remove('opacity-50', 'cursor-not-allowed');
            loading.classList.add('hidden');
        }
    });
