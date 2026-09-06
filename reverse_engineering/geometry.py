
        target_center = np.array([0.0, 0.0, 0.0], dtype=float)

        def evaluate(p):
            distance, focal, height, yaw, pitch, roll = p
            intr = CameraIntrinsics.from_focal_mm(focal, image_w, image_h)
            position, extrinsics = _camera_pose_from_params(distance, height, yaw, pitch, roll, target_center)
            camera = CameraModel(intr, extrinsics)
            projected = camera.project_points(obj)
            return projected, intr, extrinsics, position

        def residual(p):
            distance, _, height, yaw, pitch, roll = p
            projected, _, _, _ = evaluate(p)
            pred = projected[valid]
            point_r = ((pred - obs) / 4.0) * weights[:, None]
            if not np.isfinite(point_r).all():
                return np.full(point_r.size + 4, 100.0)

            extra = []
            if bbox is not None:
                bx0, by0, bx1, by1 = bbox
                pb = np.array([np.nanmin(projected[:, 0]), np.nanmin(projected[:, 1]), np.nanmax(projected[:, 0]), np.nanmax(projected[:, 1])])
                observed_w, observed_h = max(bx1 - bx0, 1.0), max(by1 - by0, 1.0)
                predicted_w, predicted_h = max(pb[2] - pb[0], 1.0), max(pb[3] - pb[1], 1.0)
                extra.extend([
                    (pb[0] - bx0) / 6.0,
                    (pb[1] - by0) / 6.0,
                    (math.log(predicted_w / observed_w)) * 6.0,
                    (math.log(predicted_h / observed_h)) * 6.0,
                ])
            # Weak regularization prevents meaningless boundary solutions while
            # keeping pose orientation free to move when image evidence supports it.
            extra.extend([
                (distance / max(base_distance, 1.0) - 1.0) * .15,
                ((height - base_height) / .8) * .10,
                (yaw / 20.0) * .06,
                (pitch / 20.0) * .06,
                (roll / 8.0) * .03,
            ])
            return np.concatenate([point_r.reshape(-1), np.asarray(extra, dtype=float)])

        results: list[PoseCandidate] = []
        yaw_seeds = (-6.0, 0.0, 6.0)
        pitch_seeds = (-6.0, 0.0, 6.0)
        roll_seeds = (-3.0, 0.0, 3.0)
        for focal in focal_seeds:
            for yaw0 in yaw_seeds:
                for pitch0 in pitch_seeds:
                    x0 = np.array([base_distance * focal / 50.0, focal, base_height, yaw0, pitch0, 0.0], dtype=float)
                    x0 = np.clip(x0, lo + 1e-4, hi - 1e-4)
                    try:
                        sol = least_squares(
                            residual, x0, bounds=(lo, hi),
                            loss="soft_l1", f_scale=5.0,
                            max_nfev=350, xtol=1e-7, ftol=1e-7, gtol=1e-7,
                        )