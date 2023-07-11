# import torch
# import torch.nn as nn
# from torch.autograd import Variable
import megengine as mge
import megengine.module as M
import megengine.functional as F

from collections import OrderedDict
import numpy as np


def summary(model, input_size, batch_size=-1, device=mge.set_default_device("xpux"), dtypes="float32"):
    result = summary_string(
        model, input_size, batch_size, device, dtypes)
    print(result)
    a = 3
    return a


def summary_string(model, input_size, batch_size=-1, device=mge.set_default_device("xpux"), dtypes="float32"):
    # if dtypes == None:
    #     dtypes = [torch.FloatTensor]*len(input_size)

    summary_str = ''

    def register_hook(module):
        def hook(module, input, output):
            # print(type(input),type(output))
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)
            # print(output[0].shape)

            m_key = "%s-%i" % (class_name, module_idx + 1)

            summary[m_key] = OrderedDict()
            print(m_key)
            if isinstance(input, tuple) and isinstance(input[0], list):
                c = 0
                for i in input[0]:
                    # print(i.shape)
                    c += i.shape[1]
                summary[m_key]["input_shape"] = [1,c,input[0][0].shape[2],input[0][0].shape[3]] 
            elif isinstance(input, tuple):
                summary[m_key]["input_shape"] = list(input[0].shape)
            else:
                summary[m_key]["input_shape"] = list(input.shape)   
            summary[m_key]["input_shape"][0] = batch_size

            if isinstance(output, (list, tuple)):
                print(type(output),len(output))
                for i in output:
                    print(i.shape)
                # summary[m_key]["output_shape"] = [
                #     [-1] + list(o.size())[1:] for o in output
                # ]
            else:
                summary[m_key]["output_shape"] = list(output.shape)
                summary[m_key]["output_shape"][0] = batch_size

            print(summary[m_key]["input_shape"])
            print(summary[m_key]["output_shape"])

            params = 0
            # if hasattr(module, "weight") and hasattr(module.weight, "size"):
            #     params += torch.prod(torch.LongTensor(list(module.weight.size())))
            #     summary[m_key]["trainable"] = module.weight.requires_grad
            # if hasattr(module, "bias") and hasattr(module.bias, "size"):
            #     params += torch.prod(torch.LongTensor(list(module.bias.size())))
            # summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, M.Sequential)
            # and not isinstance(module, nn.ModuleList)
        ):
            hooks.append(module.register_forward_hook(hook))

    # multiple inputs to the network
    # if isinstance(input_size, tuple):
    #     input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = mge.tensor(np.random.randn(1, 3, 640, 640).astype(np.float32))
    # x = mge.random.uniform(0,1,input_size).astype("float32").to(device=device)
    # x = F.expand_dims(x, 0)
    print(x.shape)

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(x)

    # remove these hooks
    for h in hooks:
        h.remove()

    summary_str += "----------------------------------------------------------------" + "\n"
    line_new = "{:>20}  {:>25} ".format(
        "Layer (type)", "Output Shape")
    summary_str += line_new + "\n"
    summary_str += "================================================================" + "\n"
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25}".format(
            layer,
            str(summary[layer]["output_shape"]),
        )
        # total_params += summary[layer]["nb_params"]

        total_output += np.prod(summary[layer]["output_shape"])
        # if "trainable" in summary[layer]:
        #     if summary[layer]["trainable"] == True:
        #         trainable_params += summary[layer]["nb_params"]
        # summary_str += line_new + "\n"

    # assume 4 bytes/number (float on cuda).
    # total_input_size = abs(np.prod(sum(input_size, ()))
    #                        * batch_size * 4. / (1024 ** 2.))
    # total_output_size = abs(2. * total_output * 4. /
    #                         (1024 ** 2.))  # x2 for gradients
    # total_params_size = abs(total_params * 4. / (1024 ** 2.))
    # total_size = total_params_size + total_output_size + total_input_size

    # summary_str += "================================================================" + "\n"
    # summary_str += "Total params: {0:,}".format(total_params) + "\n"
    # summary_str += "Trainable params: {0:,}".format(trainable_params) + "\n"
    # summary_str += "Non-trainable params: {0:,}".format(total_params -
    #                                                     trainable_params) + "\n"
    # summary_str += "----------------------------------------------------------------" + "\n"
    # summary_str += "Input size (MB): %0.2f" % total_input_size + "\n"
    # summary_str += "Forward/backward pass size (MB): %0.2f" % total_output_size + "\n"
    # summary_str += "Params size (MB): %0.2f" % total_params_size + "\n"
    # summary_str += "Estimated Total Size (MB): %0.2f" % total_size + "\n"
    # summary_str += "----------------------------------------------------------------" + "\n"
    # return summary
    return summary_str #, (total_params, trainable_params)

if __name__ == "__main__":
    image_shape = (640, 640, 3)
    anchors = np.array([[10, 13], [16, 30], [33, 23],
                        [30, 61], [62, 45], [59, 119],
                        [116, 90], [156, 198], [373, 326]]) / image_shape[0]
    anchor_masks = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=np.int8)
    anchors = np.array(anchors, dtype=np.float32)
    yolo = Yolo(num_class=80, batch_size=1, is_training=True, anchors=anchors, anchor_masks=anchor_masks)
    summary(yolo.yolov5,(3,640,640))
