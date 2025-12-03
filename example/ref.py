m = torch.hub.load("hf/dinov3", "dinov3_vits16", source="local", weights="hf/dinov3-vits16-pretrain-lvd1689m-08c60483.pth")
with torch.no_grad():
    x = m.forward_features(
        torch.as_tensor(
            np.stack(
                (
                    open("dinov23.cpp.py/example/dog.jpg").resize((1040, 1056)),
                    open("dinov23.cpp.py/example/grape.jpg").resize((1040, 1056)),
                )
            )
        ).permute(0, 3, 1, 2)
        / 255
    )["x_norm_patchtokens"]
fromarray(np.multiply(np.subtract(_ := PCA(3).fit_transform(x[0]).reshape((1056 // 16, -1, 3)), _.min((0, 1)), _), 255.9 / _.max((0, 1))).astype("B")).resize((1040, 1056)).save("dinov23.cpp.py/example/dog.jpgdinov3vit.ref", "jpeg")
fromarray(np.multiply(np.subtract(_ := PCA(3).fit_transform(x[1]).reshape((1056 // 16, -1, 3)), _.min((0, 1)), _), 255.9 / _.max((0, 1))).astype("B")).resize((1040, 1056)).save("dinov23.cpp.py/example/grape.jpgdinov3vit.ref", "jpeg")
