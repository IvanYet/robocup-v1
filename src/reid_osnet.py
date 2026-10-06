# reid_osnet.py — OSNet ReID extractor wrapper (ONNX runtime if available)
import numpy as np
import cv2

class DummyReID:
    def __init__(self, model_path=None, embed_dim=512):
        self.model_path = model_path
        self.embed_dim = embed_dim
        self.available = False

    def embed(self, img):
        # img: HxWxC BGR numpy image; return L2-normalized vector
        v = np.zeros((self.embed_dim,), dtype=np.float32)
        return v

try:
    import onnxruntime as ort
    class OSNetReID:
        def __init__(self, model_path=None, embed_dim=512):
            self.model_path = model_path
            self.embed_dim = embed_dim
            self.available = False
            if model_path is not None:
                try:
                    self.sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
                    self.available = True
                except Exception:
                    self.sess = None
                    self.available = False
        def preprocess(self, img):
            # assume 112x224 or model-specific; user should adapt
            inp = cv2.resize(img, (112, 224))
            inp = inp.astype('float32') / 255.0
            inp = np.transpose(inp, (2,0,1))[None,:,:,:]
            return inp
        def embed(self, img):
            if not self.available:
                return np.zeros((self.embed_dim,), dtype=np.float32)
            inp = self.preprocess(img)
            out = self.sess.run(None, {self.sess.get_inputs()[0].name: inp})[0]
            v = out.ravel().astype(np.float32)
            # L2 normalize
            n = np.linalg.norm(v)
            if n > 0:
                v /= n
            return v
except Exception:
    OSNetReID = DummyReID

# Provide factory

def create_reid(model_path=None, embed_dim=512):
    try:
        r = OSNetReID(model_path=model_path, embed_dim=embed_dim)
        return r
    except Exception:
        return DummyReID(model_path, embed_dim)
