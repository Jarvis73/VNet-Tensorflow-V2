from pathlib import Path

import numpy as np
import SimpleITK as sitk    # conda install simpleitk -c simpleitk


def read_scan(scan_file):
    scan = sitk.ReadImage(str(scan_file))
    direction = scan.GetDirection()
    spacing = scan.GetSpacing()
    shape = scan.GetSize()
    return scan, direction, spacing, shape


def print_data_info():
    src_dir = Path(__file__).parent / "data/train"
    for src_file in src_dir.glob("Case*.mhd"):
        if "segmentation" in str(src_file):
            continue
        seg_file = src_file.parent / (src_file.stem + "_segmentation.mhd")
        _, direction, spacing, shape = read_scan(src_file)
        _, direction2, spacing2, shape2 = read_scan(src_file)
        same = True if direction == direction2 and spacing == spacing2 and shape == shape2 else False
        print(f"{src_file.stem} - Sp: {spacing}, Size: {shape}, Di: {direction}, Match: {same}")


# print_data_info()


class DataManager(object):
    fileList = None
    gtList = None

    sitkImages = None
    sitkGT = None
    meanIntensityTrain = None

    def __init__(self, srcFolder, resultsDir, parameters):
        self.params = parameters
        self.srcFolder = Path(srcFolder)
        self.resultsDir = Path(resultsDir)

    def createImageFileList(self):
        self.fileList = list(sorted(self.srcFolder.glob("Case[0-9][0-9].mhd")))
        print(f'FILE LIST: {[str(x) for x in self.fileList]}')

    def createGTFileList(self):
        self.gtList = [f.parent / (f.stem + "_segmentation.mhd") for f in self.fileList]

    def loadImages(self):
        self.sitkImages = {}
        rescalFilt = sitk.RescaleIntensityImageFilter()
        rescalFilt.SetOutputMaximum(1)
        rescalFilt.SetOutputMinimum(0)

        stats = sitk.StatisticsImageFilter()
        m = 0.
        for f in self.fileList:
            self.sitkImages[f] = rescalFilt.Execute(
                sitk.Cast(sitk.ReadImage(str(f)), sitk.sitkFloat32))
            stats.Execute(self.sitkImages[f])
            m += stats.GetMean()
        self.meanIntensityTrain = m / len(self.sitkImages)

    def loadGT(self):
        self.sitkGT = {}
        for f in self.gtList:
            self.sitkGT[f] = sitk.Cast(sitk.ReadImage(str(f)) > 0.5, sitk.sitkFloat32)

    def loadTrainingData(self):
        self.createImageFileList()
        self.createGTFileList()
        self.loadImages()
        self.loadGT()

    def loadTestData(self):
        self.createImageFileList()
        self.loadImages()

    def getNumpyImages(self):
        dat = self.getNumpyData(self.sitkImages, sitk.sitkLinear)
        return dat

    def getNumpyGT(self):
        dat = self.getNumpyData(self.sitkGT, sitk.sitkLinear)
        for key in dat:
            dat[key] = (dat[key] > 0.5).astype(dtype=np.float32)
        return dat

    def getNumpyData(self, dat, method):
        ret = {}
        for key in dat:
            ret[key] = np.zeros([self.params['VolSize'][0], self.params['VolSize'][1], self.params['VolSize'][2]],
                                dtype=np.float32)
            img = dat[key]

            # we rotate the image according to its transformation using the direction
            # and according to the final spacing we want
            factor = np.asarray(img.GetSpacing()) / [self.params['dstRes'][0], self.params['dstRes'][1],
                                                     self.params['dstRes'][2]]

            factorSize = np.asarray(img.GetSize() * factor, dtype=float)
            newSize = np.max([factorSize, self.params['VolSize']], axis=0)
            newSize = newSize.astype(dtype=int).tolist()

            T = sitk.AffineTransform(3)
            T.SetMatrix(img.GetDirection())
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(img)
            resampler.SetOutputSpacing([self.params['dstRes'][0], self.params['dstRes'][1], self.params['dstRes'][2]])
            resampler.SetSize(newSize)
            resampler.SetInterpolator(method)
            if self.params['normDir']:
                resampler.SetTransform(T.GetInverse())
            imgResampled = resampler.Execute(img)

            imgCentroid = np.array(newSize, dtype=float) / 2.0
            imgStartPx = (imgCentroid - self.params['VolSize'] / 2.0).astype(dtype=int)

            regionExtractor = sitk.RegionOfInterestImageFilter()
            regionExtractor.SetSize(self.params['VolSize'].astype(dtype=int).tolist())
            regionExtractor.SetIndex(imgStartPx.tolist())
            imgResampledCropped = regionExtractor.Execute(imgResampled)

            ret[key] = np.transpose(sitk.GetArrayFromImage(imgResampledCropped).astype(dtype=float), [2, 1, 0])

        return ret

    def writeResultsFromNumpyLabel(self, result, key):
        img = self.sitkImages[key]
        toWrite = sitk.Image(img.GetSize()[0], img.GetSize()[1], img.GetSize()[2], sitk.sitkFloat32)

        factor = np.asarray(img.GetSpacing()) / [self.params['dstRes'][0], self.params['dstRes'][1],
                                                 self.params['dstRes'][2]]
        factorSize = np.asarray(img.GetSize() * factor, dtype=float)
        newSize = np.max([factorSize, self.params['VolSize']], axis=0)
        newSize = newSize.astype(dtype=int).tolist()

        T = sitk.AffineTransform(3)
        T.SetMatrix(img.GetDirection())

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(img)
        resampler.SetOutputSpacing([self.params['dstRes'][0], self.params['dstRes'][1], self.params['dstRes'][2]])
        resampler.SetSize(newSize)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        if self.params['normDir']:
            resampler.SetTransform(T.GetInverse())
        toWrite = resampler.Execute(toWrite)

        imgCentroid = np.array(newSize, dtype=float) / 2.0
        imgStartPx = (imgCentroid - self.params['VolSize'] / 2.0).astype(dtype=int)

        for dstX, srcX in zip(range(0, result.shape[0]),
                              range(imgStartPx[0], int(imgStartPx[0] + self.params['VolSize'][0]))):
            for dstY, srcY in zip(range(0, result.shape[1]),
                                  range(imgStartPx[1], int(imgStartPx[1] + self.params['VolSize'][1]))):
                for dstZ, srcZ in zip(range(0, result.shape[2]),
                                      range(imgStartPx[2], int(imgStartPx[2] + self.params['VolSize'][2]))):
                    try:
                        toWrite.SetPixel(int(srcX), int(srcY), int(srcZ), float(result[dstX, dstY, dstZ]))
                    except:
                        pass

        resampler.SetOutputSpacing([img.GetSpacing()[0], img.GetSpacing()[1], img.GetSpacing()[2]])
        resampler.SetSize(img.GetSize())
        if self.params['normDir']:
            resampler.SetTransform(T)
        toWrite = resampler.Execute(toWrite)

        thfilter = sitk.BinaryThresholdImageFilter()
        thfilter.SetInsideValue(1)
        thfilter.SetOutsideValue(0)
        thfilter.SetLowerThreshold(0.5)
        toWrite = thfilter.Execute(toWrite)

        # connected component analysis (better safe than sorry)
        cc = sitk.ConnectedComponentImageFilter()
        toWritecc = cc.Execute(sitk.Cast(toWrite, sitk.sitkUInt8))
        arrCC = np.transpose(sitk.GetArrayFromImage(toWritecc).astype(dtype=float), [2, 1, 0])

        lab = np.zeros(int(np.max(arrCC) + 1), dtype=float)

        for i in range(1, int(np.max(arrCC) + 1)):
            lab[i] = np.sum(arrCC == i)

        activeLab = np.argmax(lab)
        toWrite = (toWritecc == activeLab)
        toWrite = sitk.Cast(toWrite, sitk.sitkUInt8)

        writer = sitk.ImageFileWriter()
        writer.SetFileName(str(self.resultsDir / (key.stem + "_result.mhd")))
        writer.Execute(toWrite)
        print(f"Write to {self.resultsDir / (key.stem + '_result.mhd')}")


def hist_match(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image
    Arguments
    ---------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns
    -------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and counts
    _, bin_idx, s_counts = np.unique(source, return_inverse=True, return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    # interp_t_values = np.zeros_like(source,dtype=float)
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)


def produce_randomly_deformed_image(image, label, numcontrolpoints, stdDef):
    sitkImage = sitk.GetImageFromArray(image, isVector=False)
    sitklabel = sitk.GetImageFromArray(label, isVector=False)

    transfromDomainMeshSize = [numcontrolpoints] * sitkImage.GetDimension()

    tx = sitk.BSplineTransformInitializer(sitkImage, transfromDomainMeshSize)

    params = tx.GetParameters()

    paramsNp = np.asarray(params, dtype=float)
    paramsNp = paramsNp + np.random.randn(paramsNp.shape[0]) * stdDef

    paramsNp[0:int(len(params) / 3)] = 0  # remove z deformations! The resolution in z is too bad

    params = tuple(paramsNp)
    tx.SetParameters(params)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(sitkImage)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(tx)

    resampler.SetDefaultPixelValue(0)
    outimgsitk = resampler.Execute(sitkImage)
    outlabsitk = resampler.Execute(sitklabel)

    outimg = sitk.GetArrayFromImage(outimgsitk)
    outimg = outimg.astype(dtype=np.float32)

    outlbl = sitk.GetArrayFromImage(outlabsitk)
    outlbl = (outlbl > 0.5).astype(dtype=np.float32)

    return outimg, outlbl


def sitk_show(nda, title=None, margin=0.0, dpi=40):
    import matplotlib.pyplot as plt
    figsize = (1 + margin) * nda.shape[0] / dpi, (1 + margin) * nda.shape[1] / dpi

    extent = (0, nda.shape[1], nda.shape[0], 0)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])

    plt.set_cmap("gray")
    for k in range(0, nda.shape[2]):
        print("printing slice "+str(k))
        ax.imshow(np.squeeze(nda[:,:,k]),extent=extent,interpolation=None)
        plt.draw()
        plt.pause(0.1)

