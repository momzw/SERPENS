SUBDIR := src/cerpens

.PHONY: all clean build

all build clean:
	$(MAKE) -C $(SUBDIR) $@ || exit $$?