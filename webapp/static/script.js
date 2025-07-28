document.addEventListener('DOMContentLoaded', function () {
    const form = document.getElementById('churn-form');
    const nextBtn = document.getElementById('next-btn');
    const prevBtn = document.getElementById('prev-btn');
    const steps = Array.from(document.querySelectorAll('.form-step'));
    const stepperItems = Array.from(document.querySelectorAll('.step-item'));
    const formTitle = document.getElementById('form-title');
    const formSubtitle = document.getElementById('form-subtitle');

    // Sélecteurs de la modale (simplifiés)
    const modalOverlay = document.getElementById('result-modal-overlay');
    const modal = document.getElementById('result-modal');
    const modalIcon = document.getElementById('modal-icon');
    const modalTitle = document.getElementById('modal-title');
    const modalStatus = document.getElementById('modal-status');
    const modalCloseBtn = document.getElementById('modal-close-btn');

    let currentStep = 0;
    const ANIMATION_DELAY = 400;

    const stepInfo = [
        { title: "Informations sur le Compte", subtitle: "Détails du contrat et de la facturation." },
        { title: "Informations sur le Client", subtitle: "Détails démographiques du client." },
        { title: "Services Souscrits", subtitle: "Quels services le client utilise-t-il ?" }
    ];

    nextBtn.addEventListener('click', () => {
        if (validateStep(currentStep)) {
            currentStep < steps.length - 1 ? changeStep('next') : submitForm();
        }
    });

    prevBtn.addEventListener('click', () => {
        if (currentStep > 0) changeStep('prev');
    });

    function changeStep(direction) {
        const currentStepElement = steps[currentStep];
        currentStepElement.classList.add('slide-out');
        setTimeout(() => {
            currentStepElement.classList.remove('active', 'slide-out');
            currentStep = (direction === 'next') ? currentStep + 1 : currentStep - 1;
            steps[currentStep].classList.add('active');
            updateUI();
        }, ANIMATION_DELAY);
    }
    
    function updateUI() {
        stepperItems.forEach((item, index) => {
            item.classList.remove('active', 'completed');
            if (index < currentStep) item.classList.add('completed');
            else if (index === currentStep) item.classList.add('active');
        });
        
        formTitle.textContent = stepInfo[currentStep].title;
        formSubtitle.textContent = stepInfo[currentStep].subtitle;
        prevBtn.style.display = currentStep > 0 ? 'inline-flex' : 'none';
        nextBtn.innerHTML = (currentStep === steps.length - 1) 
            ? 'Analyser le Risque <i class="fas fa-robot"></i>' 
            : 'Suivant <i class="fas fa-arrow-right"></i>';
    }

    function validateStep(stepIndex) {
        let isValid = true;
        const fields = steps[stepIndex].querySelectorAll('input[required], select[required]');
        fields.forEach(field => {
            field.classList.remove('invalid');
            if (!field.value.trim()) {
                isValid = false;
                field.classList.add('invalid');
            }
        });
        return isValid;
    }

    function submitForm() {
        nextBtn.disabled = true;
        nextBtn.innerHTML = '<span class="loading-spinner"></span>Analyse...';

        const data = Object.fromEntries(new FormData(form).entries());

        fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data),
        })
        .then(response => response.json())
        .then(result => {
            if (result.error) throw new Error(result.error);
            // On passe uniquement la prédiction à la fonction
            showModal(result.prediction);
        })
        .catch(error => {
            console.error('Erreur:', error);
            showModal('api_error');
        })
        .finally(() => {
            nextBtn.disabled = false;
            updateUI();
        });
    }

    // Fonction showModal simplifiée (sans la confiance)
    function showModal(prediction) {
        modal.classList.remove('success', 'error');

        const resultTypes = {
            1: { class: 'error', icon: 'fa-exclamation-triangle', title: 'Risque de Churn Élevé', status: "<strong>Statut :</strong> Ce client est susceptible de résilier son contrat. Une action de rétention proactive est fortement recommandée." },
            0: { class: 'success', icon: 'fa-check', title: 'Faible Risque de Churn', status: "<strong>Statut :</strong> Ce client est fidèle. Continuez à offrir une excellente expérience pour maintenir sa satisfaction." },
            'api_error': { class: 'error', icon: 'fa-server', title: 'Erreur de Connexion', status: "<strong>Statut :</strong> Impossible de contacter le serveur de prédiction. Veuillez réessayer plus tard." }
        };

        const config = resultTypes[prediction];
        modal.classList.add(config.class);
        modalIcon.className = `modal-icon ${config.class}`;
        modalIcon.innerHTML = `<i class="fas ${config.icon}"></i>`;
        modalTitle.textContent = config.title;
        modalStatus.innerHTML = config.status;
        
        modalOverlay.classList.add('open');
    }
    
    function hideModal() {
        modalOverlay.classList.remove('open');
    }

    modalCloseBtn.addEventListener('click', hideModal);
    modalOverlay.addEventListener('click', e => { if (e.target === modalOverlay) hideModal(); });

    updateUI();
    
    const style = document.createElement('style');
    style.innerHTML = `.loading-spinner { display: inline-block; width: 16px; height: 16px; border: 2px solid rgba(255, 255, 255, 0.3); border-radius: 50%; border-top-color: white; animation: spin 1s ease-in-out infinite; } @keyframes spin { to { transform: rotate(360deg); } }`;
    document.head.appendChild(style);
});