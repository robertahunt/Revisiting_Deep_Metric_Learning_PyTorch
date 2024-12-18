import datasets.cub200
import datasets.cars196
import datasets.stanford_online_products
import datasets.rove
import datasets.inaturalist_2021_mini


def select(dataset, opt, data_path):
    if 'cub200' in dataset:
        return cub200.Give(opt, data_path)

    if 'cars196' in dataset:
        return cars196.Give(opt, data_path)

    if 'online_products' in dataset:
        return stanford_online_products.Give(opt, data_path)

    if 'rove' in dataset:
        return rove.Give(opt, data_path)

    if 'inaturalist' in dataset:
        return inaturalist_2021_mini.Give(opt, data_path)

    raise NotImplementedError('A dataset for {} is currently not implemented.\n\
                               Currently available are : cub200, cars196, online_products, rove, and inaturalist_2021_mini!'.format(dataset))
