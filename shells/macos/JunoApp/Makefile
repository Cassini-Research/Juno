# Assembles Juno.app from SwiftPM release binaries (shells/macos).
# Invoked by scripts/package_juno_macos_app.sh or Xcode Legacy target.
MK_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
SPM_ROOT := $(abspath $(MK_DIR)/..)
REPO_ROOT := $(abspath $(SPM_ROOT)/../..)
BUILD ?= $(SPM_ROOT)/.build/release
SWIFT_BUILD_FLAGS ?=
JUNO_APP_VERSION ?=
JUNO_BUILD_NUMBER ?=
JUNO_OTA_FEED_URL ?=
JUNO_OTA_PUBLIC_ED_KEY ?=
JUNO_OTA_CHANNEL ?=
JUNO_OTA_DISABLED ?=
JUNO_OTA_ALLOW_INSECURE_FEED ?=
JUNO_OTA_AUTOMATIC_CHECKS ?=
JUNO_OTA_AUTOMATIC_DOWNLOADS ?=
JUNO_OTA_SCHEDULED_INTERVAL ?=
# Default output under JunoApp/build; override with DIST_APP=...
DIST_APP ?= $(abspath $(MK_DIR)/build/Juno.app)
# When building from Xcode Legacy target, prefer the build products directory.
ifdef BUILT_PRODUCTS_DIR
ifneq ($(strip $(BUILT_PRODUCTS_DIR)),)
DIST_APP := $(BUILT_PRODUCTS_DIR)/Juno.app
endif
endif

HELPERS := juno-paste juno-hotkey juno-textmon juno-capability juno-host
JUNO_RESOURCE_BUNDLE := JunoShell_JunoShell.bundle

.PHONY: swift-release app build clean

swift-release:
	# The --package-path form crashes swift-package (dyld: BuildServerProtocol)
	# on current CommandLineTools; building from inside the package works.
	cd "$(SPM_ROOT)" && swift build $(SWIFT_BUILD_FLAGS) -c release || \
	  { test -x "$(SPM_ROOT)/.build/release/Juno" && \
	    echo "WARNING: swift toolchain aborted; reusing existing release binaries"; }

# Xcode invokes `make build` by default for Legacy targets.
build: app

app: swift-release
	@echo "→ $(DIST_APP)"
	rm -rf "$(DIST_APP)"
	mkdir -p "$(DIST_APP)/Contents/MacOS" "$(DIST_APP)/Contents/Resources" "$(DIST_APP)/Contents/Frameworks"
	cp "$(BUILD)/Juno" "$(DIST_APP)/Contents/MacOS/Juno"
	@SPARKLE_FW="$$(find "$(SPM_ROOT)/.build" -path "*/Sparkle.framework" -type d 2>/dev/null | head -n 1)"; \
	if [ -n "$$SPARKLE_FW" ]; then \
	  rsync -a "$$SPARKLE_FW" "$(DIST_APP)/Contents/Frameworks/"; \
	  echo "Bundled Sparkle.framework"; \
	else \
	  echo "warning: missing Sparkle.framework under $(SPM_ROOT)/.build"; \
	fi
	@if [ -d "$(BUILD)/$(JUNO_RESOURCE_BUNDLE)" ]; then \
	  cp -R "$(BUILD)/$(JUNO_RESOURCE_BUNDLE)" "$(DIST_APP)/Contents/Resources/$(JUNO_RESOURCE_BUNDLE)"; \
	else echo "warning: missing $(BUILD)/$(JUNO_RESOURCE_BUNDLE)"; fi
	@for h in $(HELPERS); do \
	  if [ -f "$(BUILD)/$$h" ]; then cp "$(BUILD)/$$h" "$(DIST_APP)/Contents/MacOS/$$h"; \
	  else echo "warning: missing $(BUILD)/$$h"; fi; \
	done
	chmod +x "$(DIST_APP)/Contents/MacOS/Juno"
	@for h in $(HELPERS); do chmod +x "$(DIST_APP)/Contents/MacOS/$$h" 2>/dev/null || true; done
	@ENGINE_SRC="$(MK_DIR)/../bundle_resources/engine"; \
	if [ -f "$$ENGINE_SRC/run_engine.sh" ]; then \
	  mkdir -p "$(DIST_APP)/Contents/Resources/engine"; \
	  cp "$$ENGINE_SRC/run_engine.sh" "$(DIST_APP)/Contents/Resources/engine/run_engine.sh"; \
	  chmod +x "$(DIST_APP)/Contents/Resources/engine/run_engine.sh"; \
	  echo "Bundled Resources/engine/run_engine.sh (populate with scripts/build_juno_engine_bundle.sh)"; \
	fi
	@mkdir -p "$(DIST_APP)/Contents/Resources/scripts"; \
	cp "$(REPO_ROOT)/scripts/install_juno_launchd.sh" "$(DIST_APP)/Contents/Resources/scripts/install_juno_launchd.sh"; \
	chmod +x "$(DIST_APP)/Contents/Resources/scripts/install_juno_launchd.sh"
	cp "$(MK_DIR)/Info.plist" "$(DIST_APP)/Contents/Info.plist"
	@JUNO_APP_VERSION="$(JUNO_APP_VERSION)" \
	  JUNO_BUILD_NUMBER="$(JUNO_BUILD_NUMBER)" \
	  JUNO_OTA_FEED_URL="$(JUNO_OTA_FEED_URL)" \
	  JUNO_OTA_PUBLIC_ED_KEY="$(JUNO_OTA_PUBLIC_ED_KEY)" \
	  JUNO_OTA_CHANNEL="$(JUNO_OTA_CHANNEL)" \
	  JUNO_OTA_DISABLED="$(JUNO_OTA_DISABLED)" \
	  JUNO_OTA_ALLOW_INSECURE_FEED="$(JUNO_OTA_ALLOW_INSECURE_FEED)" \
	  JUNO_OTA_AUTOMATIC_CHECKS="$(JUNO_OTA_AUTOMATIC_CHECKS)" \
	  JUNO_OTA_AUTOMATIC_DOWNLOADS="$(JUNO_OTA_AUTOMATIC_DOWNLOADS)" \
	  JUNO_OTA_SCHEDULED_INTERVAL="$(JUNO_OTA_SCHEDULED_INTERVAL)" \
	  python3 "$(REPO_ROOT)/scripts/configure_juno_macos_plist.py" "$(DIST_APP)/Contents/Info.plist" --from-env
	# Generate Brand Kit AppIcon.icns into the bundle for Finder/Dock.
	swift "$(SPM_ROOT)/../../scripts/generate_juno_app_icon_icns.swift" "$(DIST_APP)/Contents/Resources/AppIcon.icns"
	printf 'APPL????' > "$(DIST_APP)/Contents/PkgInfo"
	# Re-sign adhoc with entitlements. SwiftPM's release output is already
	# adhoc + hardened runtime, but with NO entitlements file — under that
	# combination AVCaptureDevice silently fails (no TCC prompt, no row in
	# Privacy → Microphone). Re-sign here so every local build is correct.
	@ENTS="$(MK_DIR)/Juno.entitlements"; \
	if [ ! -f "$$ENTS" ]; then echo "missing entitlements: $$ENTS" >&2; exit 2; fi; \
	"$(REPO_ROOT)/scripts/sign_juno_macos_app.sh" --app "$(DIST_APP)" --identity - --entitlements "$$ENTS"
	@echo "Built $(DIST_APP)"

clean:
	rm -rf "$(MK_DIR)/build"
