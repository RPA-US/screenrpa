document.addEventListener('DOMContentLoaded', function() {
    console.log('El formulario se está procesando, script cargado');
    const form = document.querySelector('#processDiscoveryForm');
    const modelTypeSelect = document.querySelector('[name="model_type"]');
    const textWeightInput = document.querySelector('[name="text_weight"]');
    const imageWeightInput = document.querySelector('[name="image_weight"]');
    const configurationsField = document.querySelector('[name="configurations"]');
    const textColumnSelect = document.querySelector('[name="text_column"]');
    const removeLoopsCheckbox = document.querySelector('[name="remove_loops"]');

    console.log(modelTypeSelect, textWeightInput, imageWeightInput, configurationsField, textColumnSelect, removeLoopsCheckbox);

    function updateConfigurations() {
        const configurations = {
            model_type: modelTypeSelect.value,
            clustering_type: document.querySelector('[name="clustering_type"]').value,
            labeling: document.querySelector('[name="labeling"]').value,
            use_pca: document.querySelector('[name="use_pca"]').checked,
            n_components: document.querySelector('[name="n_components"]').value,
            show_dendrogram: document.querySelector('[name="show_dendrogram"]').checked,
            remove_loops: removeLoopsCheckbox.checked,
        };

        if (modelTypeSelect.value === 'clip') {
            configurations.text_weight = textWeightInput.value;
            configurations.image_weight = imageWeightInput.value;
            configurations.text_column = textColumnSelect.value;
        }
        
        console.log('Configuraciones actualizadas:', configurations);
        configurationsField.value = JSON.stringify(configurations);
    }

    form.querySelectorAll('input, select').forEach(element => {
        element.addEventListener('change', updateConfigurations);
    });

    modelTypeSelect.addEventListener('change', function() {
        toggleWeightFields();
        updateConfigurations(); 
    });

    toggleWeightFields();
    updateConfigurations();
});

function toggleWeightFields() {
    const modelTypeSelect = document.querySelector('[name="model_type"]');
    const weightsRow = document.querySelector('#weightsRow');
    weightsRow.style.display = modelTypeSelect.value === 'clip' ? 'block' : 'none';
}
