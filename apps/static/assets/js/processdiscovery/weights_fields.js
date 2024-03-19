document.addEventListener('DOMContentLoaded', function() {
    console.log('El formulario se está procesasando, script cargado')
    const form = document.querySelector('#processDiscoveryForm');
    console.log('Form', form)
    const modelTypeSelect = document.querySelector('[name="model_type"]');
    const textWeightInput = document.querySelector('[name="text_weight"]');
    const imageWeightInput = document.querySelector('[name="image_weight"]');
    const configurationsField = document.querySelector('[name="configurations"]');
    console.log(modelTypeSelect, textWeightInput, imageWeightInput, configurationsField)
    
    function updateConfigurations() {
        const configurations = {
            model_type: modelTypeSelect.value,
            clustering_type: document.querySelector('[name="clustering_type"]').value,
            labeling: document.querySelector('[name="labeling"]').value,
            use_pca: document.querySelector('[name="use_pca"]').checked,
            n_components: document.querySelector('[name="n_components"]').value,
            show_dendrogram: document.querySelector('[name="show_dendrogram"]').checked,
        };

        if(modelTypeSelect.value === 'clip') {
            configurations.text_weight = textWeightInput ? textWeightInput.value : '';
            configurations.image_weight = imageWeightInput ? imageWeightInput.value : '';
        }
        console.log('Configuraciones actualizadas:', configurations.value)
        configurationsField.value = JSON.stringify(configurations);
    }

    form.querySelectorAll('input, select').forEach(element => {
        element.addEventListener('change', updateConfigurations);
    });

    modelTypeSelect.addEventListener('change', function() {
        const weightsRow = document.querySelector('#weightsRow');
        weightsRow.style.display = this.value === 'clip' ? 'block' : 'none';
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