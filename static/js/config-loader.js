/**
 * Configuration Loader for Nail Detection System
 * Loads configuration from .env_config file and provides fallback defaults
 */

class ConfigLoader {
  constructor() {
    this.config = {};
    this.defaultConfig = this.getDefaultConfig();
    this.loadConfig();
  }

  /**
   * Default configuration fallback
   */
  getDefaultConfig() {
    return {
      // Hand Detection Settings
      handDetection: {
        maxHands: 2,
        modelComplexity: 1,
        minDetectionConfidence: 0.6,
        minTrackingConfidence: 0.5,
      },

      // Nail Visibility Detection Methods
      nailVisibility: {
        method: "combined", // 'hand_confidence', 'landmark_stability', 'custom_geometric', 'combined'
        handConfidence: {
          enabled: true,
          threshold: 0.5,
        },
        landmarkStability: {
          enabled: false,
          threshold: 0.05,
          historyLength: 3,
          minFrames: 2,
        },
        customGeometric: {
          enabled: true,
          visibilityThreshold: 0.6,
          angleThreshold: 0.8,
          distanceThreshold: 0.05,
        },
      },

      // Finger Selection Settings
      fingerSelection: {
        all: true,
        thumb: true,
        index: true,
        middle: true,
        ring: true,
        pinky: true,
      },

      // Nail Rendering Settings
      nailRendering: {
        opacity: 0.9,
        baseScale: 1.0,
        customScale: 1.0,
        autoScale: false,
        rotationOffset: 0,
        smoothing: 0.2,
        placementRatio: 1.1,
        mobileIndexOffset: 0.05,
      },

      // Nail Stabilization Settings
      nailStabilization: {
        enabled: true,
        movementThreshold: 40,
        transitionSpeed: 8,
        maxTransitionDistance: 50,
        useEasing: true,
        easingType: "easeOutCubic",
      },

      // Movement Tolerance Settings
      movementTolerance: {
        enabled: true,
        duration: 1000,
        confidenceBoost: 0.2,
      },

      // Rotation Stabilization Settings
      rotationStabilization: {
        enabled: true,
        threshold: 0.3,
        historyLength: 5,
        smoothing: 0.3,
      },

      // Finger-Specific Scaling Settings
      fingerScaling: {
        thumb: { min: 0.02, max: 0.12, base: 0.7 },
        index: { min: 0.02, max: 0.15, base: 0.78 },
        middle: { min: 0.02, max: 0.18, base: 0.91 },
        ring: { min: 0.02, max: 0.16, base: 0.83 },
        pinky: { min: 0.02, max: 0.14, base: 0.73 },
      },

      // TIP-DIP Distance Thresholds
      tipDip: {
        threshold: 0.05,
        minDistance: 0.02,
        maxDistance: 0.15,
      },

      // Debug and Visualization Settings
      debug: {
        showHandLandmarks: false,
        showFingerNumbers: true,
        showDebugInfo: false,
        showStabilizationStatus: true,
      },

      // Performance Settings
      performance: {
        maxFps: 60,
        skipFrames: false,
        frameSkipInterval: 1,
      },

      // Error Handling Settings
      errorHandling: {
        maxRetries: 3,
        retryDelay: 1000,
        roiRecovery: true,
        videoValidation: true,
      },

      // Mobile Optimization Settings
      mobile: {
        detectionEnabled: true,
        touchDetection: true,
        screenWidthThreshold: 768,
        performanceMode: false,
      },
    };
  }

  /**
   * Load configuration from .env_config file
   */
  async loadConfig() {
    try {
      const response = await fetch("/.env_config");
      if (response.ok) {
        const configText = await response.text();
        this.parseConfigText(configText);
      } else {
        console.warn("Config file not found, using default configuration");
        this.config = { ...this.defaultConfig };
      }
    } catch (error) {
      console.warn(
        "Failed to load config file, using default configuration:",
        error
      );
      this.config = { ...this.defaultConfig };
    }
  }

  /**
   * Parse configuration text from .env file
   */
  parseConfigText(configText) {
    this.config = { ...this.defaultConfig };

    const lines = configText.split("\n");

    for (const line of lines) {
      const trimmedLine = line.trim();

      // Skip empty lines and comments
      if (!trimmedLine || trimmedLine.startsWith("#")) {
        continue;
      }

      const [key, value] = trimmedLine.split("=");
      if (key && value !== undefined) {
        this.setConfigValue(key.trim(), value.trim());
      }
    }
  }

  /**
   * Set configuration value based on key path
   */
  setConfigValue(key, value) {
    // Convert string values to appropriate types
    let convertedValue = value;

    if (value === "true") convertedValue = true;
    else if (value === "false") convertedValue = false;
    else if (!isNaN(value) && value !== "") convertedValue = parseFloat(value);

    // Map configuration keys to nested structure
    const keyMap = {
      // Hand Detection Settings
      HAND_DETECTION_MAX_HANDS: "handDetection.maxHands",
      HAND_DETECTION_MODEL_COMPLEXITY: "handDetection.modelComplexity",
      HAND_DETECTION_MIN_DETECTION_CONFIDENCE:
        "handDetection.minDetectionConfidence",
      HAND_DETECTION_MIN_TRACKING_CONFIDENCE:
        "handDetection.minTrackingConfidence",

      // Nail Visibility Detection Methods
      NAIL_VISIBILITY_METHOD: "nailVisibility.method",
      HAND_CONFIDENCE_THRESHOLD: "nailVisibility.handConfidence.threshold",
      HAND_CONFIDENCE_ENABLED: "nailVisibility.handConfidence.enabled",
      LANDMARK_STABILITY_ENABLED: "nailVisibility.landmarkStability.enabled",
      LANDMARK_STABILITY_THRESHOLD:
        "nailVisibility.landmarkStability.threshold",
      LANDMARK_STABILITY_HISTORY_LENGTH:
        "nailVisibility.landmarkStability.historyLength",
      LANDMARK_STABILITY_MIN_FRAMES:
        "nailVisibility.landmarkStability.minFrames",
      CUSTOM_GEOMETRIC_ENABLED: "nailVisibility.customGeometric.enabled",
      CUSTOM_GEOMETRIC_VISIBILITY_THRESHOLD:
        "nailVisibility.customGeometric.visibilityThreshold",
      CUSTOM_GEOMETRIC_ANGLE_THRESHOLD:
        "nailVisibility.customGeometric.angleThreshold",
      CUSTOM_GEOMETRIC_DISTANCE_THRESHOLD:
        "nailVisibility.customGeometric.distanceThreshold",

      // Finger Selection Settings
      FINGER_SELECTION_ALL: "fingerSelection.all",
      FINGER_SELECTION_THUMB: "fingerSelection.thumb",
      FINGER_SELECTION_INDEX: "fingerSelection.index",
      FINGER_SELECTION_MIDDLE: "fingerSelection.middle",
      FINGER_SELECTION_RING: "fingerSelection.ring",
      FINGER_SELECTION_PINKY: "fingerSelection.pinky",

      // Nail Rendering Settings
      NAIL_OPACITY: "nailRendering.opacity",
      NAIL_BASE_SCALE: "nailRendering.baseScale",
      NAIL_CUSTOM_SCALE: "nailRendering.customScale",
      NAIL_AUTO_SCALE: "nailRendering.autoScale",
      NAIL_ROTATION_OFFSET: "nailRendering.rotationOffset",
      NAIL_SMOOTHING: "nailRendering.smoothing",
      NAIL_PLACEMENT_RATIO: "nailRendering.placementRatio",
      NAIL_MOBILE_INDEX_OFFSET: "nailRendering.mobileIndexOffset",

      // Nail Stabilization Settings
      NAIL_STABILIZATION_ENABLED: "nailStabilization.enabled",
      NAIL_STABILIZATION_MOVEMENT_THRESHOLD:
        "nailStabilization.movementThreshold",
      NAIL_STABILIZATION_TRANSITION_SPEED: "nailStabilization.transitionSpeed",
      NAIL_STABILIZATION_MAX_TRANSITION_DISTANCE:
        "nailStabilization.maxTransitionDistance",
      NAIL_STABILIZATION_USE_EASING: "nailStabilization.useEasing",
      NAIL_STABILIZATION_EASING_TYPE: "nailStabilization.easingType",
      MOVEMENT_TOLERANCE_ENABLED: "movementTolerance.enabled",
      MOVEMENT_TOLERANCE_DURATION: "movementTolerance.duration",
      MOVEMENT_TOLERANCE_CONFIDENCE_BOOST: "movementTolerance.confidenceBoost",
      ROTATION_STABILIZATION_ENABLED: "rotationStabilization.enabled",
      ROTATION_STABILIZATION_THRESHOLD: "rotationStabilization.threshold",
      ROTATION_STABILIZATION_HISTORY_LENGTH:
        "rotationStabilization.historyLength",
      ROTATION_STABILIZATION_SMOOTHING: "rotationStabilization.smoothing",

      // Debug Settings
      DEBUG_SHOW_HAND_LANDMARKS: "debug.showHandLandmarks",
      DEBUG_SHOW_FINGER_NUMBERS: "debug.showFingerNumbers",
      DEBUG_SHOW_DEBUG_INFO: "debug.showDebugInfo",
      DEBUG_SHOW_STABILIZATION_STATUS: "debug.showStabilizationStatus",
    };

    const configPath = keyMap[key];
    if (configPath) {
      this.setNestedValue(this.config, configPath, convertedValue);
    }
  }

  /**
   * Set nested object value using dot notation
   */
  setNestedValue(obj, path, value) {
    const keys = path.split(".");
    let current = obj;

    for (let i = 0; i < keys.length - 1; i++) {
      if (!current[keys[i]]) {
        current[keys[i]] = {};
      }
      current = current[keys[i]];
    }

    current[keys[keys.length - 1]] = value;
  }

  /**
   * Get configuration value
   */
  get(key) {
    return this.getNestedValue(this.config, key);
  }

  /**
   * Get nested object value using dot notation
   */
  getNestedValue(obj, path) {
    return path.split(".").reduce((current, key) => current?.[key], obj);
  }

  /**
   * Get all configuration
   */
  getAll() {
    return this.config;
  }

  /**
   * Update configuration
   */
  update(newConfig) {
    this.config = { ...this.config, ...newConfig };
  }
}

// Make ConfigLoader available globally
window.ConfigLoader = ConfigLoader;
