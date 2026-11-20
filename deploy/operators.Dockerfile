ARG BASE_IMAGE=quay.io/ascend/vllm-ascend:v0.30.0
FROM ${BASE_IMAGE} AS compiler
ARG ANYINFER_TARGET_SOC
ARG MAX_JOBS
WORKDIR /opt/anyinfer-build
COPY operators operators
COPY operator-source.sha256 compile-operators.sh ./
RUN --network=none bash compile-operators.sh

FROM ${BASE_IMAGE}
ARG ANYINFER_TARGET_SOC
ARG OPERATOR_SOURCE_SHA256
WORKDIR /opt/anyinfer
RUN python3 -m pip uninstall -y vllm-ascend
COPY --from=compiler /opt/anyinfer/operators /opt/anyinfer/operators
COPY operators/smoke_hc_post.py /opt/anyinfer/operators/smoke_hc_post.py
COPY operator_integrity.py /opt/anyinfer/operators/operator_integrity.py
COPY calibrate.py /opt/anyinfer/operators/calibrate.py
ENV PYTHONPATH="/opt/anyinfer/operators/python:${PYTHONPATH}"
ENV PYTHONDONTWRITEBYTECODE=1
RUN --mount=type=bind,source=overlay,target=/tmp/anyinfer-overlay python3 /tmp/anyinfer-overlay/apply_overlay.py /tmp/anyinfer-overlay/files
RUN python3 /opt/anyinfer/operators/operator_integrity.py record "$ANYINFER_TARGET_SOC" "$OPERATOR_SOURCE_SHA256"
