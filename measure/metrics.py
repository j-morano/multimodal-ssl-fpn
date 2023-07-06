import numpy as np
from medpy.metric.binary import hd, hd95
from torch.nn import functional as F

from measure import ssim



class Metrics(object):
    def __init__(self):
        self.accumulator = []

    def calculate_batch(self, _ground: dict, _predict: dict) -> np.ndarray:
        raise NotImplementedError

    def update(self, ground, predict):
        result = self.calculate_batch(ground,predict)
        self.accumulator.extend(result.tolist())

    def get(self):
        return np.nanmean(self.accumulator)

    def reset(self):
        self.accumulator = []


class Dice(Metrics):
    def __init__(self, output_key: str, target_key: str, slice:int = 0):
        super(Dice, self).__init__()
        self.output_key=output_key
        self.target_key=target_key
        self.slice = slice

    def calculate_batch(self, ground:dict, predict:dict) -> np.ndarray:
        pred = predict[self.output_key].detach()
        gr = ground[self.target_key].detach()

        pred_slice = self.slice
        assert (gr[:,self.slice].shape == pred[:,self.slice].shape)

        N = pred.size(0)

        pred = (pred[:,pred_slice]>0.5).float().view(N,-1)
        gr = (gr[:,self.slice]>0.5).float().view(N,-1)

        numerator = (pred * gr).sum(dim=1).cpu().numpy()
        denominator = (pred + gr).sum(dim=1).cpu().numpy()

        r = 2 * numerator / denominator
        r[denominator==0.] = 1

        return r


class BCE(Metrics):
    """Binary Cross Entropy metric."""
    def __init__(self, output_key: str, target_key: str, bg_weight=1, slice: int=0):
        super().__init__()
        self.output_key = output_key
        self.target_key = target_key
        self.bg_weight=bg_weight
        self.counter = 0
        self.slice = slice

    def calculate_batch(self, ground:dict, predict:dict) -> np.ndarray:
        pred = predict[self.output_key].detach()
        gr = ground[self.target_key].detach()

        assert (gr[:,self.slice].shape == pred[:,self.slice].shape)

        pred = pred[:,self.slice].view(pred.size(0),-1)
        gr = gr[:,self.slice].view(gr.size(0),-1)

        bce = F.binary_cross_entropy(pred, gr, reduction='none')
        bce = bce.mean(dim=1).cpu().numpy()

        return bce


class Hausdorff(Metrics):
    def __init__(self, output_key=0, target_key=0, slice:int=0):
        super(Hausdorff, self).__init__()
        self.output_key=output_key
        self.target_key=target_key
        self.slice=slice

    def calculate_batch(self, ground: dict, predict: dict) -> np.ndarray:
        pred = predict[self.output_key].detach()
        gr = ground[self.target_key].detach()

        assert (gr[:,self.slice].shape == pred[:,self.slice].shape)

        pred = (pred>0.5).int().cpu().numpy()
        gr = (gr>0.5).int().cpu().numpy()

        result = []

        for n in range(pred.shape[0]):
            p = pred[n,self.slice].astype(np.uint8)
            g = gr[n,self.slice].astype(np.uint8)

            if (p.sum() == 0) or (g.sum() == 0):
                result.append(np.nan)
                continue
            else:

                try:
                    spacing = ground['spacing'][n].cpu().numpy()
                except KeyError:
                    affine = ground['affine'][n].cpu().numpy()
                    spacing = np.sqrt(
                        (affine ** 2).sum(axis=0)
                    )[:len(p.shape)] # type: ignore
                try:
                    r = hd(p[:,0],g[:,0],voxelspacing=spacing[[0,2]])
                    result.append(r)
                except RuntimeError as E:
                    print("Hausdorff:RuntimeError: "+ str(E))
                    pass

        return np.array(result)


class Hausdorff95(Metrics):
    def __init__(self, output_key: str, target_key: str, slice:int=0):
        super(Hausdorff95, self).__init__()
        self.output_key=output_key
        self.target_key=target_key
        self.slice=slice

    def calculate_batch(self, ground: dict, predict: dict) -> np.ndarray:
        pred = predict[self.output_key].detach()
        gr = ground[self.target_key].detach()

        assert (gr[:,self.slice].shape == pred[:,self.slice].shape)

        pred = (pred>0.5).int().cpu().numpy()
        gr = (gr>0.5).int().cpu().numpy()

        result = []

        for n in range(pred.shape[0]):
            p = pred[n,self.slice].astype(np.uint8)
            g = gr[n,self.slice].astype(np.uint8)

            if (p.sum() == 0) or (g.sum() == 0):
                result.append(np.nan)
                continue
            else:

                try:
                    spacing = ground['spacing'][n].cpu().numpy()
                except KeyError:
                    affine = ground['affine'][n].cpu().numpy()
                    spacing = np.sqrt(
                        (affine ** 2).sum(axis=0)
                    )[:len(p.shape)] # type: ignore
                try:
                    r = hd95(p[:,0],g[:,0],voxelspacing=spacing[[0,2]],connectivity=3)
                    result.append(r)
                except RuntimeError as E:
                    print("Hausdorff95:RuntimeError: "+ str(E))
                    pass

        return np.array(result)


class SSIM(Metrics):
    def __init__(self, output_key: str, target_key: str, slice:int=0):
        super().__init__()
        self.output_key=output_key
        self.target_key=target_key
        self.slice=slice
        self.ssim = ssim.SSIM(loss=False)

    def calculate_batch(self, ground: dict, predict: dict) -> np.ndarray:
        pred = predict[self.output_key].detach().cpu()
        gr = ground[self.target_key].detach().cpu()

        assert (gr[:,self.slice].shape == pred[:,self.slice].shape)

        p = pred[:,self.slice:self.slice+1,:,0]
        g = gr[:,self.slice:self.slice+1,:,0]

        return np.array([self.ssim(p,g).numpy()])
