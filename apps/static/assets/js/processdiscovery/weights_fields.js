let isFormProcessed = false;
document.addEventListener('DOMContentLoaded', function() {
    const form = document.querySelector('#processDiscoveryForm');

    const modelTypeSelect = document.querySelector('[name="model_type"]');
    const weightsRow = document.querySelector('#weightsRow');

    function toggleWeightFields() {
        if (modelTypeSelect.value === 'clip') {
            weightsRow.style.display = 'block';
        } else {
            weightsRow.style.display = 'none';
        }
    }

    modelTypeSelect.addEventListener('change', toggleWeightFields);
    toggleWeightFields();

    form.addEventListener('submit', function(event) {
        if (!isFormProcessed) {
            event.preventDefault(); 
            const configurations = {
                model_type: document.querySelector('[name="model_type"]').value,
                model_weights: document.querySelector('[name="model_weights"]').value || '',
                clustering_type: document.querySelector('[name="clustering_type"]').value,
                labeling: document.querySelector('[name="labeling"]').value,
                use_pca: document.querySelector('[name="use_pca"]').checked,
                n_components: document.querySelector('[name="n_components"]').value,
                show_dendrogram: document.querySelector('[name="show_dendrogram"]').checked,
            };

            const configurationsField = document.querySelector('[name="configurations"]');
            configurationsField.value = JSON.stringify(configurations);

            console.log('Configurations updated:', configurationsField.value);

            isFormProcessed = true; 
            form.submit(); 
        }
    });
});
