.PHONY: default build clean remote-push-all

default:
	@echo "Please specify a target to build"

clean:
	rm -rf output
	mkdir output

SOURCE_DIR=.
TARGET_DIR=output
build: clean
	@aigen generate -d $(SOURCE_DIR)
	@find $(SOURCE_DIR) -type f -name '*.json' -exec cp {} $(TARGET_DIR) \;
	rm $(TARGET_DIR)/README.json
	rm $(TARGET_DIR)/template.json

remote-push-all: build
	scp $(TARGET_DIR)/*.json alex@192.168.50.150:/home/alex/docker/genaitutor/wwwroot/data/.
