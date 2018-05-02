from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf

import mushroom_data


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int)
parser.add_argument('--train_steps', default=1000, type=int)

def main(argv):
    args = parser.parse_args(argv[1:])

    (train_x, train_y), (test_x, test_y) = mushroom_data.load_data()

    mushroom_feature_columns = []
    for key in train_x.keys():
        print('\n' + key)
        mushroom_feature_columns.append(tf.feature_column.numeric_column(key=key))

    classifier = tf.estimator.DNNClassifier(
        feature_columns=mushroom_feature_columns,
        hidden_units=[10, 10],
        n_classes=2)

    classifier.train(
        input_fn=lambda:mushroom_data.train_input_fn(train_x, train_y,
                                                 args.batch_size),
        steps=args.train_steps)

    eval_result = classifier.evaluate(
        input_fn=lambda:mushroom_data.eval_input_fn(test_x, test_y,
                                                args.batch_size))

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    expected = ['0', '1']
    predict_x = {
        'cap.shape_b': [0.0966212211, 0.0124839949],
        'cap.shape_c': [0, 0.0006402049],
        'cap.shape_f': [0.3778897451, 0.4010883483],
        'cap.shape_k': [0.0542382928, 0.1507682458],
        'cap.shape_s': [0.0080023711, 0],
        'cap.shape_x': [0.4632483699, 0.4350192061],
        'cap.surface_f': [0.3648488441, 0.1920614597],
        'cap.surface_g': [0, 0.0009603073],
        'cap.surface_s': [0.2685240071, 0.3581946223],
        'cap.surface_y': [0.3666271488, 0.4487836108],
        'cap.color_b': [0.012151749, 0.032010243],
        'cap.color_c': [0.006520451, 0.003201024],
        'cap.color_e': [0.144339063, 0.221190781],
        'cap.color_g': [0.246591583, 0.208386684],
        'cap.color_n': [0.302608180, 0.257682458],
        'cap.color_p': [0.013337285, 0.021766965],
        'cap.color_r': [0.004445762, 0],
        'cap.color_u': [0.004149378, 0],
        'cap.color_w': [0.170124481, 0.082266325],
        'cap.color_y': [0.095732069, 0.173495519],
        'bruises_f': [0.3426200, 0.8370679],
        'bruises_t': [0.6573800, 0.1629321],
        'odor_a': [0.100474215, 0],
        'odor_c': [0, 0.048335467],
        'odor_f': [0, 0.552816901],
        'odor_l': [0.094546532, 0],
        'odor_m': [0, 0.009282971],
        'odor_n': [0.804979253, 0.030729834],
        'odor_p': [0, 0.064980794],
        'odor_s': [0, 0.143405890],
        'odor_y': [0, 0.150448143],
        'gill.attachment_a': [0.045050385, 0.004801536],
        'gill.attachment_f': [0.954949615, 0.995198464],
        'gill.spacing_c': [0.71665679, 0.97151088],
        'gill.spacing_w': [0.28334321, 0.02848912],
        'gill.size_b': [0.9309425, 0.4395006],
        'gill.size_n': [0.0690575, 0.5604994],
        'gill.color_b': [0, 0.436299616],
        'gill.color_e': [0.023117961, 0],
        'gill.color_g': [0.058091286, 0.128681178],
        'gill.color_h': [0.049199763, 0.139884763],
        'gill.color_k': [0.080616479, 0.016005122],
        'gill.color_n': [0.224066390, 0.028489117],
        'gill.color_o': [0.015115590, 0],
        'gill.color_p': [0.203912270, 0.164532650],
        'gill.color_r': [0, 0.006402049],
        'gill.color_u': [0.107291049, 0.011203585],
        'gill.color_w': [0.223770006, 0.063380282],
        'gill.color_y': [0.014819206, 0.005121639],
        'stalk.shape_e': [0.3858921, 0.4875160],
        'stalk.shape_t': [0.6141079, 0.5124840],
        'stalk.surface.above.ring_f': [0.099585062, 0.038412292],
        'stalk.surface.above.ring_k': [0.032009484, 0.569462228],
        'stalk.surface.above.ring_s': [0.864848844, 0.390204866],
        'stalk.surface.above.ring_y': [0.003556609, 0.001920615],
        'stalk.surface.above.ring_y': [0.003556609, 0.001920615],
        'stalk.surface.below.ring_f': [0.10877297, 0.03841229],
        'stalk.surface.below.ring_k': [0.03230587, 0.55025608],
        'stalk.surface.below.ring_s': [0.80705394, 0.39308579],
        'stalk.surface.below.ring_y': [0.05186722, 0.01824584],
        'stalk.color.above.ring_b': [0, 0.108514725],
        'stalk.color.above.ring_c': [0, 0.009282971],
        'stalk.color.above.ring_e': [0.023710729, 0],
        'stalk.color.above.ring_g': [0.138411381, 0],
        'stalk.color.above.ring_n': [0.003556609, 0.115877081],
        'stalk.color.above.ring_o': [0.045050385, 0],
        'stalk.color.above.ring_p': [0.138707765, 0.329385403],
        'stalk.color.above.ring_w': [0.650563130, 0.435019206],
        'stalk.color.above.ring_y': [0, 0.001920615],
        'stalk.color.below.ring_b': [0, 0.112355954],
        'stalk.color.below.ring_c': [0, 0.009282971],
        'stalk.color.below.ring_e': [0.024303497, 0],
        'stalk.color.below.ring_g': [0.135743924, 0],
        'stalk.color.below.ring_n': [0.015115590, 0.111715749],
        'stalk.color.below.ring_o': [0.045050385, 0],
        'stalk.color.below.ring_p': [0.136336692, 0.333546735],
        'stalk.color.below.ring_w': [0.643449911, 0.426696543],
        'stalk.color.below.ring_y': [0, 0.006402049],
        'veil.color_n': [0.022228809, 0],
        'veil.color_o': [0.022821577, 0],
        'veil.color_w': [0.954949615, 0.998079385],
        'veil.color_y': [0, 0.001920615],
        'ring.number_n': [0, 0.009282971],
        'ring.number_o': [0.876407825, 0.970870679],
        'ring.number_t': [0.123592175, 0.019846351],
        'ring.type_e': [0.241256669, 0.445262484],
        'ring.type_f': [0.011855365, 0],
        'ring.type_l': [0, 0.334186940],
        'ring.type_n': [0, 0.009282971],
        'ring.type_p': [0.746887967, 0.211267606],
        'spore.print.color_b': [0.01126260, 0],
        'spore.print.color_h': [0.01185536, 0.41037132],
        'spore.print.color_k': [0.39330172, 0.05633803],
        'spore.print.color_n': [0.41375222, 0.05697823],
        'spore.print.color_o': [0.01126260, 0],
        'spore.print.color_r': [0, 0.01984635],
        'spore.print.color_u': [0.01215175, 0],
        'spore.print.color_w': [0.13515116, 0.45646607],
        'spore.print.color_y': [0.01126260, 0],
        'population_a': [0.09187908, 0],
        'population_c': [0.07083580, 0.01312420],
        'population_n': [0.09365738, 0],
        'population_s': [0.21102549, 0.09635083],
        'population_v': [0.28275044, 0.72247119],
        'population_y': [0.24985181, 0.16805378],
        'habitat_d': [0.445465323, 0.325544174],
        'habitat_g': [0.335210433, 0.192701665],
        'habitat_l': [0.056609366, 0.149487836],
        'habitat_m': [0.061647896, 0.009603073],
        'habitat_p': [0.031713100, 0.253201024],
        'habitat_u': [0.022228809, 0.069462228],
        'habitat_w': [0.047125074, 0]
    }

    predictions = classifier.predict(
        input_fn=lambda:mushroom_data.eval_input_fn(predict_x,
                                                labels=None,
                                                batch_size=args.batch_size))

    template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')

    for pred_dict, expec in zip(predictions, expected):
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print(template.format(mushroom_data.CLASS[class_id],
                              100 * probability, expec))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
