IMAGE=breast-cancer-prediction:latest
CONTAINER=breast-cancer-prediction

.PHONY: train build run stop logs test report

train:
\tpython -m model.train

build:
\tdocker build -t $(IMAGE) .

run:
\tdocker run --rm -d -p 8000:8000 --name $(CONTAINER) -v $$PWD/logs:/app/logs $(IMAGE)

stop:
\t-docker stop $(CONTAINER)

logs:
\tdocker logs -f $(CONTAINER)

test:
\tpytest -q || true

report:
\tpython -m reports.make_report
