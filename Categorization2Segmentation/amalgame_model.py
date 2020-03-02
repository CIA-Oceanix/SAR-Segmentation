from keras.layers import Input, Concatenate
from keras.models import Model, load_model
from keras import backend as K


def format_name(model_name):
    while ' ' in model_name:
        model_name = model_name.replace(' ', '_')
    return model_name


def import_model(categorization_model_filename, segmentation_model_filenames):
    categorization_model = load_model(categorization_model_filename)
    input_shape = categorization_model.input_shape
    input_layer = Input(input_shape)

    print('l16, input_layer:', input_layer)

    assert categorization_model.output_shape[-1] == len(segmentation_model_filenames), \
        f"You have {len(segmentation_model_filenames)} segmentation_models but your categorization has " \
            f"{categorization_model.output_shape[-1]} outputs"

    segmentation_models = []
    segmentation_outputs = []
    for model_filename in segmentation_model_filenames:
        print('l26, loading model', model_filename)
        segmentation_model = load_model(model_filename)
        print(segmentation_model.input_shape, input_shape)
        segmentation_model.name = format_name(segmentation_model.name)
        print(segmentation_model.summary())

        segmentation_outputs.append(segmentation_model(input_layer))
        segmentation_models.append(segmentation_model)

    segmentation_output = Concatenate(axis=-1)(segmentation_outputs)

    categorization_output = categorization_model(input_layer)
    output = K.dot(segmentation_output, categorization_output)

    heatmap_model = Model(input_layer, output, name="heatmap_model")
    heatmap_model.compile()
    heatmap_model.summary()
    heatmap_model.submodels = segmentation_models + [categorization_model]
    return heatmap_model
