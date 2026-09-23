# Research executor image (Arch Linux base).
#
# Pinned dated base snapshot for reproducibility. Provides pacman-managed
# system tools (python, pip, uv, poetry, ffmpeg, libsndfile, JDK, PlantUML)
# plus a uv-installed research Python stack in /opt/executor-venv.
FROM archlinux:base-20260920.0.596911

# Refresh the keyring against current signing state, then install the system
# toolchain. The pinned base image means repo versions resolve at build time;
# both package managers record their resolved sets for auditing. PlantUML
# pulls a Java runtime; the full JDK is included so agents can also compile.
RUN pacman-key --init \
    && pacman -Sy --noconfirm archlinux-keyring \
    && pacman -Su --noconfirm \
    && pacman -S --noconfirm --needed \
        python \
        python-pip \
        uv \
        python-poetry \
        ffmpeg \
        libsndfile \
        git \
        ca-certificates \
        jdk-openjdk \
        plantuml \
    && pacman -Scc --noconfirm \
    && rm -rf /var/cache/pacman/pkg/* /var/lib/pacman/sync/*

# Non-root execution user, matching the service's exec uid/gid (65532).
RUN groupadd -g 65532 onyx-exec \
    && useradd -u 65532 -g 65532 -d /workspace -s /bin/sh onyx-exec

# Research Python stack. The venv sits on PATH so `python`/`pip` resolve to it;
# sessions additionally create their own /workspace/.venv at runtime.
COPY requirements.in /tmp/requirements.in
ENV VIRTUAL_ENV=/opt/executor-venv
RUN python -m venv $VIRTUAL_ENV \
    && uv pip install --python "$VIRTUAL_ENV/bin/python" --upgrade pip setuptools wheel \
    && uv pip install --python "$VIRTUAL_ENV/bin/python" -r /tmp/requirements.in \
    && "$VIRTUAL_ENV/bin/python" -c "import numpy, scipy, pandas, matplotlib, PIL, cv2, librosa, soundfile, requests, httpx, openpyxl; print('research executor: package check OK')" \
    && uv pip freeze --python "$VIRTUAL_ENV/bin/python" > /opt/executor-requirements.lock \
    && pacman -Q > /opt/pacman-packages.txt \
    && rm /tmp/requirements.in

ENV PATH=/opt/executor-venv/bin:${PATH}

# The service mounts a tmpfs (ephemeral) or volume (session) at /workspace.
WORKDIR /workspace

CMD ["python3"]
