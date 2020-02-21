from keras.layers import Input, Concatenate
from keras.models import Model, load_model
from keras import backend as K


def import_model(categorization_model_filename, segmentation_model_filenames):
    segmentation_models = [load_model(segmentation_model_filename)
                           for segmentation_model_filename in segmentation_model_filenames]
    categorization_model = load_model(categorization_model_filename)
    input_shape = categorization_model.input_shape

    assert categorization_model.output_shape[-1] == len(segmentation_models), \
        f"You have {len(segmentation_models)} segmentation_models but your categorization has " \
            f"{categorization_model.output_shape[-1]} outputs"
    for i, segmentation_model in enumerate(categorization_model):
        assert input_shape[0] == segmentation_model.input_shape[0], \
            f"Segmentation model {i} has an input shape of f{segmentation_model.input_shape} whereas the categorizer" \
                f"accept an input of shape {input_shape}"

    input_layer = Input(input_shape)
    segmentation_outputs = [segmentation_model(input_layer) for segmentation_model in segmentation_models]
    segmentation_output = Concatenate(axis=-1)(segmentation_outputs)

    categorization_output = categorization_model(input_layer)
    output = K.dot(segmentation_output, categorization_output)

    heatmap_model = Model(input_layer, output, name="heatmap_model")
    heatmap_model.compile()
    heatmap_model.summary()
    return heatmap_model
