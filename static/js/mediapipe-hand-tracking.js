class MediaPipeHandTracker {
    constructor() {
        this.video = null;
        this.canvas = null;
        this.ctx = null;
        this.hands = null;
        this.isInitialized = false;
        this.isRunning = false;
        this.segmentedImage = null;
        this.segmentedImageUrl = null;
        this.rafId = null;
        
        // Configuration loader
        this.configLoader = new ConfigLoader();
        this.config = {};
        
        // Settings
        this.settings = {
            opacity: 0.9,
            baseScale: 1.0,
            customScale: 1.0,  // Custom scale that works independently of auto scale
            smoothing: 0.2,
            showHandLandmarks: false,  // Disabled by default
            showFingerNumbers: true,
            showDebugInfo: false,
            autoScale: false,
            rotationOffset: 0,  // User-configurable rotation offset in radians
            minDistance: 0.3,
            maxDistance: 1.0,
            // Mobile-specific settings
            mobileIndexOffset: 0.05,  // Offset for index finger on mobile (5% of finger length)
            nailPlacementRatio: 1.1,  // Place nail at 110% of finger length instead of tip
            // Dynamic scaling based on TIP-DIP distance
            tipDipThreshold: 0.05,  // Threshold for TIP-DIP distance (adjustable)
            minTipDipDistance: 0.02,
            maxTipDipDistance: 0.15,
            // Finger-specific scaling thresholds (increased by 160%)
            fingerScaling: {
                thumb: { min: 0.02, max: 0.12, base: 0.70 },    // 0.25 * 2.6 = 0.65
                index: { min: 0.02, max: 0.15, base: 0.78 },    // 0.3 * 2.6 = 0.78
                middle: { min: 0.02, max: 0.18, base: 0.91 },   // 0.35 * 2.6 = 0.91
                ring: { min: 0.02, max: 0.16, base: 0.83 },     // 0.32 * 2.6 = 0.83
                pinky: { min: 0.02, max: 0.14, base: 0.73 }     // 0.28 * 2.6 = 0.73
            },
            // Visibility detection settings
            visibilityThreshold: 0.3,  // Threshold for nail visibility detection
            stabilityThreshold: 0.02,  // Threshold for landmark stability
            
            // Nail mask stabilization settings
            nailStabilization: {
                enabled: true,  // Enable nail mask stabilization
                movementThreshold: 40,  // Pixels - minimum movement to update position
                transitionSpeed: 2,  // 0-1, higher = faster transition (0.15 = smooth)
                maxTransitionDistance: 50,  // Pixels - max distance to transition smoothly
                useEasing: true,  // Use easing function for natural movement
                easingType: 'easeOutCubic'  // Easing function type
            }
        };
        
        // Finger selection
        this.fingerSelection = {
            all: true,
            thumb: true,
            index: true,
            middle: true,
            ring: true,
            pinky: true
        };
        
        // Hand data
        this.handData = {
            isDetected: false,
            confidence: 0,
            handedness: null,
            landmarks: [],
            center: { x: 0, y: 0 },
            distance: 0
        };
        
        // MediaPipe finger mapping (21 landmarks)
        this.FINGER_MAP = {
            thumb: [1, 2, 3, 4],      // Thumb landmarks
            index: [5, 6, 7, 8],      // Index finger landmarks
            middle: [9, 10, 11, 12],  // Middle finger landmarks
            ring: [13, 14, 15, 16],   // Ring finger landmarks
            pinky: [17, 18, 19, 20]   // Pinky finger landmarks
        };
        
        // TIP and DIP landmark indices for each finger
        this.TIP_DIP_MAP = {
            thumb: { tip: 4, dip: 3, pip: 2, mcp: 1 },
            index: { tip: 8, dip: 6, pip: 5, mcp: 5 },
            middle: { tip: 12, dip: 10, pip: 9, mcp: 9 },
            ring: { tip: 16, dip: 14, pip: 13, mcp: 13 },
            pinky: { tip: 20, dip: 18, pip: 17, mcp: 17 }
        };
        
        // Palm center landmark (landmark 0)
        this.PALM_CENTER = 0;
        
        // Stability tracking for error handling
        this.landmarkHistory = {
            positions: [],
            maxHistory: 5,
            stability: {}
        };
        
        // Nail positions for each finger
        this.nailData = {
            positions: {},
            scales: {},
            rotations: {}
        };
        
        // Nail mask stabilization data
        this.nailStabilization = {
            lastPositions: {},  // Last stable positions for each finger
            targetPositions: {},  // Target positions for smooth transitions
            isTransitioning: {},  // Whether each finger is currently transitioning
            transitionStartTime: {},  // When transition started for each finger
            transitionStartPosition: {},  // Starting position for transition
            // New: DIP stabilization for angle-based shaking
            lastDipPositions: {},  // Last stable DIP positions for each finger
            targetDipPositions: {},  // Target DIP positions for smooth transitions
            isDipTransitioning: {},  // Whether DIP is currently transitioning
            dipTransitionStartTime: {},  // When DIP transition started for each finger
            dipTransitionStartPosition: {},  // Starting DIP position for transition
            // NEW: Rotation stabilization to prevent shaking
            lastRotations: {},  // Last stable rotations for each finger
            targetRotations: {},  // Target rotations for smooth transitions
            isRotationTransitioning: {},  // Whether rotation is currently transitioning
            rotationTransitionStartTime: {},  // When rotation transition started
            rotationTransitionStartAngle: {},  // Starting rotation angle
            rotationHistory: {},  // Rotation history for smoothing
            maxRotationHistory: 5  // Maximum rotation history length
        };
        
        // Debug info
        this.debugInfo = {
            lastDetectionTime: 0,
            handCount: 0,
            fps: 0
        };
        
        this.lastFrameTime = 0;
    }
    
    // Easing functions for smooth transitions
    easingFunctions = {
        linear: (t) => t,
        easeOutCubic: (t) => 1 - Math.pow(1 - t, 3),
        easeInOutCubic: (t) => t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2,
        easeOutQuart: (t) => 1 - Math.pow(1 - t, 4),
        easeOutExpo: (t) => t === 1 ? 1 : 1 - Math.pow(2, -10 * t)
    };
    
    async init(video, canvas, segmentedImageUrl) {
        try {
            this.video = video;
            this.canvas = canvas;
            this.ctx = canvas.getContext('2d');
            this.segmentedImageUrl = segmentedImageUrl;
            
            // Load configuration
            await this.configLoader.loadConfig();
            this.config = this.configLoader.getAll();
            this.updateSettingsFromConfig();
            
            // Load segmented image
            await this.loadSegmentedImage();
            
            // Initialize MediaPipe Hands
            await this.initializeMediaPipe();
            
            this.isInitialized = true;
            console.log('MediaPipe hand tracker initialized with config:', this.config);
            
        } catch (error) {
            console.error('MediaPipe hand tracker initialization failed:', error);
            throw error;
        }
    }
    
    async loadSegmentedImage() {
        return new Promise((resolve, reject) => {
            this.segmentedImage = new Image();
            this.segmentedImage.crossOrigin = 'anonymous';
            this.segmentedImage.onload = () => {
                console.log('Segmented image loaded for MediaPipe hand tracking');
                resolve();
            };
            this.segmentedImage.onerror = (error) => {
                console.error('Failed to load segmented image:', error);
                reject(error);
            };
            this.segmentedImage.src = this.segmentedImageUrl;
        });
    }
    
    updateSettingsFromConfig() {
        // Update settings from configuration
        if (this.config.nailRendering) {
            this.settings.opacity = this.config.nailRendering.opacity;
            this.settings.baseScale = this.config.nailRendering.baseScale;
            this.settings.customScale = this.config.nailRendering.customScale;
            this.settings.autoScale = this.config.nailRendering.autoScale;
            this.settings.rotationOffset = this.config.nailRendering.rotationOffset;
            this.settings.smoothing = this.config.nailRendering.smoothing;
            this.settings.nailPlacementRatio = this.config.nailRendering.placementRatio;
            this.settings.mobileIndexOffset = this.config.nailRendering.mobileIndexOffset;
        }
        
        if (this.config.debug) {
            this.settings.showHandLandmarks = this.config.debug.showHandLandmarks;
            this.settings.showFingerNumbers = this.config.debug.showFingerNumbers;
            this.settings.showDebugInfo = this.config.debug.showDebugInfo;
        }
        
        if (this.config.nailStabilization) {
            this.settings.nailStabilization = { ...this.settings.nailStabilization, ...this.config.nailStabilization };
        }
        
        if (this.config.fingerScaling) {
            this.settings.fingerScaling = { ...this.settings.fingerScaling, ...this.config.fingerScaling };
        }
        
        if (this.config.fingerSelection) {
            this.fingerSelection = { ...this.fingerSelection, ...this.config.fingerSelection };
        }
    }

    async initializeMediaPipe() {
        // Check if MediaPipe is available
        if (typeof Hands === 'undefined') {
            throw new Error('MediaPipe Hands not available');
        }
        
        try {
          this.hands = new Hands({
            locateFile: (file) =>
              `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
          });

          // Use configuration for MediaPipe settings
          const handDetectionConfig = this.config.handDetection || {};
          this.hands.setOptions({
            maxNumHands: handDetectionConfig.maxHands || 2,
            modelComplexity: handDetectionConfig.modelComplexity || 1,
            minDetectionConfidence:
              handDetectionConfig.minDetectionConfidence || 0.6,
            minTrackingConfidence:
              handDetectionConfig.minTrackingConfidence || 0.5,
          });

          this.hands.onResults(this.onResults.bind(this));

          // Add error handler for MediaPipe
          this.hands.onError = (error) => {
            console.error("MediaPipe error:", error);
            // Don't throw here to avoid breaking the loop
          };
        } catch (error) {
            console.error('Failed to initialize MediaPipe:', error);
            throw error;
        }
    }
    
    start() {
        if (!this.isInitialized || this.isRunning) return;
        
        this.isRunning = true;
        this.sendLoop();
        console.log('MediaPipe hand tracker started');
    }
    
    stop() {
        this.isRunning = false;
        if (this.rafId) {
            cancelAnimationFrame(this.rafId);
            this.rafId = null;
        }
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        console.log('MediaPipe hand tracker stopped');
    }
    
    async sendLoop() {
        if (!this.isRunning) return;
        
        try {
            // Validate video dimensions before processing
            if (!this.isVideoValid()) {
                console.warn('Video dimensions invalid, skipping frame');
                this.rafId = requestAnimationFrame(() => this.sendLoop());
                return;
            }
            
            // Resize canvas to match video
            if (this.canvas.width !== this.video.videoWidth || this.canvas.height !== this.video.videoHeight) {
                this.canvas.width = this.video.videoWidth;
                this.canvas.height = this.video.videoHeight;
            }
            
            // Send frame to MediaPipe
            await this.hands.send({ image: this.video });
            
        } catch (error) {
            console.warn('MediaPipe send error:', error);
            
            // If it's a ROI error, try to recover by reinitializing
            if (error.message && error.message.includes('ROI width and height must be > 0')) {
                console.warn('ROI error detected, attempting recovery...');
                await this.handleROIError();
            }
        }
        
        this.rafId = requestAnimationFrame(() => this.sendLoop());
    }
    
    isVideoValid() {
        // Check if video is ready and has valid dimensions
        if (!this.video || this.video.readyState < 2) {
            return false;
        }
        
        const width = this.video.videoWidth;
        const height = this.video.videoHeight;
        
        // Validate dimensions
        if (!width || !height || width <= 0 || height <= 0) {
            console.warn(`Invalid video dimensions: ${width}x${height}`);
            return false;
        }
        
        // Check if video is not too small (common issue on mobile)
        if (width < 64 || height < 64) {
            console.warn(`Video too small: ${width}x${height}`);
            return false;
        }
        
        return true;
    }
    
    async handleROIError() {
        try {
            console.log('Attempting to recover from ROI error...');
            
            // Stop current processing
            this.stop();
            
            // Wait a bit for video to stabilize
            await new Promise(resolve => setTimeout(resolve, 1000));
            
            // Reinitialize MediaPipe with retry
            await this.initializeMediaPipeWithRetry();
            
            // Restart if video is valid
            if (this.isVideoValid()) {
                this.start();
                console.log('Successfully recovered from ROI error');
            } else {
                console.error('Video still invalid after recovery attempt');
            }
        } catch (error) {
            console.error('Failed to recover from ROI error:', error);
        }
    }
    
    async initializeMediaPipeWithRetry(maxRetries = 3) {
        let lastError;
        
        for (let attempt = 1; attempt <= maxRetries; attempt++) {
            try {
                console.log(`MediaPipe initialization attempt ${attempt}/${maxRetries}`);
                await this.initializeMediaPipe();
                console.log('MediaPipe initialized successfully');
                return;
            } catch (error) {
                lastError = error;
                console.warn(`MediaPipe initialization attempt ${attempt} failed:`, error);
                
                if (attempt < maxRetries) {
                    // Exponential backoff: wait 1s, 2s, 4s
                    const delay = Math.pow(2, attempt - 1) * 1000;
                    console.log(`Waiting ${delay}ms before retry...`);
                    await new Promise(resolve => setTimeout(resolve, delay));
                }
            }
        }
        
        throw lastError || new Error('MediaPipe initialization failed after all retries');
    }
    
    onResults(results) {
        // Update FPS
        const now = Date.now();
        if (this.lastFrameTime > 0) {
            this.debugInfo.fps = Math.round(1000 / (now - this.lastFrameTime));
        }
        this.lastFrameTime = now;
        
        // Clear canvas
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Process results
        if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
            this.debugInfo.handCount = results.multiHandLandmarks.length;
            this.debugInfo.lastDetectionTime = now;
            
            // Process each detected hand
            for (let i = 0; i < results.multiHandLandmarks.length; i++) {
              const landmarks = results.multiHandLandmarks[i];
              const handedness =
                results.multiHandedness && results.multiHandedness[i]
                  ? results.multiHandedness[i].label.toLowerCase()
                  : "unknown";

              // Get hand confidence from MediaPipe results
              const handConfidence =
                results.multiHandedness && results.multiHandedness[i]
                  ? results.multiHandedness[i].score || 0.8
                  : 0.8;

              // Update hand data with actual confidence
              this.handData.isDetected = true;
              this.handData.confidence = handConfidence;
              this.handData.handedness = handedness;
              this.handData.landmarks = landmarks;

              // Calculate hand center
              this.calculateHandCenter(landmarks);

              // Draw hand landmarks if enabled
              if (this.settings.showHandLandmarks) {
                this.drawHandLandmarks(landmarks);
              }

              // Draw nail overlays with improved finger tracking
              this.drawNailOverlays(landmarks, handedness);
            }
        } else {
            // No hands detected
            this.handData.isDetected = false;
            this.handData.confidence = 0;
            this.debugInfo.handCount = 0;
        }
        
        // Draw debug info if enabled
        if (this.settings.showDebugInfo) {
            this.drawDebugInfo();
        }
        
    }
    
    calculateHandCenter(landmarks) {
        let sumX = 0, sumY = 0;
        landmarks.forEach(landmark => {
            sumX += landmark.x;
            sumY += landmark.y;
        });
        
        this.handData.center = {
            x: (sumX / landmarks.length) * this.canvas.width,
            y: (sumY / landmarks.length) * this.canvas.height
        };
    }
    
    drawHandLandmarks(landmarks) {
        // Draw all 21 hand landmarks with different colors for different fingers
        landmarks.forEach((landmark, index) => {
            const x = landmark.x * this.canvas.width;
            const y = landmark.y * this.canvas.height;
            
            // Color code different fingers
            let color = 'rgba(0, 255, 0, 0.8)'; // Default green
            let pointSize = 4;
            
            // Thumb (0-4)
            if (index >= 0 && index <= 4) {
                color = 'rgba(255, 0, 0, 0.8)'; // Red
            }
            // Index finger (5-8)
            else if (index >= 5 && index <= 8) {
                color = 'rgba(0, 0, 255, 0.8)'; // Blue
            }
            // Middle finger (9-12)
            else if (index >= 9 && index <= 12) {
                color = 'rgba(255, 255, 0, 0.8)'; // Yellow
            }
            // Ring finger (13-16)
            else if (index >= 13 && index <= 16) {
                color = 'rgba(255, 0, 255, 0.8)'; // Magenta
            }
            // Pinky (17-20)
            else if (index >= 17 && index <= 20) {
                color = 'rgba(0, 255, 255, 0.8)'; // Cyan
            }
            
            // Draw the landmark point
            this.ctx.fillStyle = color;
            this.ctx.beginPath();
            this.ctx.arc(x, y, pointSize, 0, 2 * Math.PI);
            this.ctx.fill();
            
            // Draw white border for better visibility
            this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.9)';
            this.ctx.lineWidth = 1;
            this.ctx.stroke();
            
            // Draw landmark number
            if (this.settings.showFingerNumbers) {
                this.ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
                this.ctx.font = 'bold 12px Arial';
                this.ctx.textAlign = 'center';
                this.ctx.fillText(index.toString(), x, y - 12);
            }
        });
        
        // Draw finger connections
        this.drawFingerConnections(landmarks);
    }
    
    drawFingerConnections(landmarks) {
        this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.6)';
        this.ctx.lineWidth = 2;
        
        // Thumb connections
        this.drawConnection(landmarks, 0, 1);
        this.drawConnection(landmarks, 1, 2);
        this.drawConnection(landmarks, 2, 3);
        this.drawConnection(landmarks, 3, 4);
        
        // Index finger connections
        this.drawConnection(landmarks, 5, 6);
        this.drawConnection(landmarks, 6, 7);
        this.drawConnection(landmarks, 7, 8);
        
        // Middle finger connections
        this.drawConnection(landmarks, 9, 10);
        this.drawConnection(landmarks, 10, 11);
        this.drawConnection(landmarks, 11, 12);
        
        // Ring finger connections
        this.drawConnection(landmarks, 13, 14);
        this.drawConnection(landmarks, 14, 15);
        this.drawConnection(landmarks, 15, 16);
        
        // Pinky connections
        this.drawConnection(landmarks, 17, 18);
        this.drawConnection(landmarks, 18, 19);
        this.drawConnection(landmarks, 19, 20);
        
        // Palm connections
        this.drawConnection(landmarks, 0, 5);
        this.drawConnection(landmarks, 5, 9);
        this.drawConnection(landmarks, 9, 13);
        this.drawConnection(landmarks, 13, 17);
        this.drawConnection(landmarks, 0, 17);
    }
    
    drawConnection(landmarks, startIndex, endIndex) {
        if (startIndex < landmarks.length && endIndex < landmarks.length) {
            const start = landmarks[startIndex];
            const end = landmarks[endIndex];
            
            this.ctx.beginPath();
            this.ctx.moveTo(start.x * this.canvas.width, start.y * this.canvas.height);
            this.ctx.lineTo(end.x * this.canvas.width, end.y * this.canvas.height);
            this.ctx.stroke();
        }
    }
    
    drawNailOverlays(landmarks, handedness) {
        if (!this.segmentedImage || !this.segmentedImage.complete) return;
        
        // Update landmark stability tracking
        this.updateLandmarkStability(landmarks);
        
        // Draw nails for each selected finger
        Object.keys(this.FINGER_MAP).forEach(finger => {
            // Debug logging
            if (this.settings.showDebugInfo) {
                console.log(`Finger ${finger}: all=${this.fingerSelection.all}, individual=${this.fingerSelection[finger]}`);
            }
            
            // Show nail if "all" is selected OR if individual finger is selected
            if (this.fingerSelection.all || this.fingerSelection[finger]) {
                const tipDipData = this.TIP_DIP_MAP[finger];
                
                if (this.isValidLandmarkSet(landmarks, tipDipData)) {
                  const tipLandmark = landmarks[tipDipData.tip];
                  const dipLandmark = landmarks[tipDipData.dip];
                  const pipLandmark = landmarks[tipDipData.pip];
                  const palmCenter = landmarks[this.PALM_CENTER];

                  // Store current finger context for visibility detection
                  this.currentFinger = finger;

                  // Check if nail is visible using new methods
                  if (
                    this.isNailVisible(
                      tipLandmark,
                      dipLandmark,
                      pipLandmark,
                      palmCenter,
                      handedness
                    )
                  ) {
                    // Calculate dynamic scale and rotation based on TIP-DIP
                    const { scale, rotation } =
                      this.calculateDynamicScaleAndRotation(
                        tipLandmark,
                        dipLandmark,
                        pipLandmark,
                        finger
                      );

                    // Check if landmarks are stable enough (if stability method is enabled)
                    const stabilityConfig =
                      this.config.nailVisibility?.landmarkStability;
                    const shouldCheckStability =
                      stabilityConfig?.enabled || false;

                    if (
                      !shouldCheckStability ||
                      this.isLandmarkStable(tipDipData.tip)
                    ) {
                      this.drawNailAtLandmark(
                        tipLandmark,
                        finger,
                        scale,
                        rotation
                      );
                    }
                  }
                }
            }
        });
    }
    
    calculateNailPosition(tipLandmark, finger) {
        // Get the current hand landmarks for this finger
        const tipDipData = this.TIP_DIP_MAP[finger];
        const landmarks = this.handData.landmarks;
        
        if (!landmarks || landmarks.length === 0) {
            // Fallback to tip position if no landmarks available
            return { x: tipLandmark.x, y: tipLandmark.y };
        }
        
        // Get finger landmarks
        const tip = landmarks[tipDipData.tip];
        const dip = landmarks[tipDipData.dip];
        const pip = landmarks[tipDipData.pip];
        const mcp = landmarks[tipDipData.mcp];
        
        // Calculate finger direction vector (from MCP to TIP)
        const fingerDirection = {
            x: tip.x - mcp.x,
            y: tip.y - mcp.y
        };
        
        // Calculate finger length
        const fingerLength = Math.sqrt(fingerDirection.x * fingerDirection.x + fingerDirection.y * fingerDirection.y);
        
        // Calculate position at 75% of finger length from MCP
        const placementRatio = this.settings.nailPlacementRatio; // 0.75
        const nailX = mcp.x + (fingerDirection.x * placementRatio);
        const nailY = mcp.y + (fingerDirection.y * placementRatio);
        
        // Apply mobile-specific offset for index finger
        let finalX = nailX;
        let finalY = nailY;
        
        if (finger === 'index') {
            // Detect if we're on mobile (small screen or touch device)
            const isMobile = window.innerWidth <= 768 || 'ontouchstart' in window;
            
            if (isMobile) {
                // Apply mobile offset perpendicular to finger direction
                const perpendicularX = -fingerDirection.y / fingerLength;
                const perpendicularY = fingerDirection.x / fingerLength;
                
                const offsetAmount = this.settings.mobileIndexOffset * fingerLength;
                finalX += perpendicularX * offsetAmount;
                finalY += perpendicularY * offsetAmount;
            }
        }
        
        return { x: finalX, y: finalY };
    }
    
    // Calculate stabilized nail position with movement threshold and smooth transitions
    calculateStabilizedNailPosition(landmark, finger) {
        const nailPosition = this.calculateNailPosition(landmark, finger);
        const currentX = nailPosition.x * this.canvas.width;
        const currentY = nailPosition.y * this.canvas.height;
        
        // If stabilization is disabled, return current position
        if (!this.settings.nailStabilization.enabled) {
            return { x: currentX, y: currentY };
        }
        
        const stabilization = this.settings.nailStabilization;
        const stabilizationData = this.nailStabilization;
        
        // Initialize stabilization data for this finger if not exists
        if (!stabilizationData.lastPositions[finger]) {
            stabilizationData.lastPositions[finger] = { x: currentX, y: currentY };
            stabilizationData.targetPositions[finger] = { x: currentX, y: currentY };
            stabilizationData.isTransitioning[finger] = false;
            return { x: currentX, y: currentY };
        }
        
        const lastPos = stabilizationData.lastPositions[finger];
        const targetPos = stabilizationData.targetPositions[finger];
        
        // Calculate movement distance
        const dx = currentX - lastPos.x;
        const dy = currentY - lastPos.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        
        // If movement is below threshold, keep last position (stabilize)
        if (distance < stabilization.movementThreshold) {
            return { x: lastPos.x, y: lastPos.y };
        }
        
        // Movement exceeds threshold - start or continue transition
        const now = Date.now();
        
        if (!stabilizationData.isTransitioning[finger]) {
            // Start new transition
            stabilizationData.isTransitioning[finger] = true;
            stabilizationData.transitionStartTime[finger] = now;
            stabilizationData.transitionStartPosition[finger] = { x: lastPos.x, y: lastPos.y };
            stabilizationData.targetPositions[finger] = { x: currentX, y: currentY };
        } else {
            // Update target position if movement is significant
            const targetDx = currentX - targetPos.x;
            const targetDy = currentY - targetPos.y;
            const targetDistance = Math.sqrt(targetDx * targetDx + targetDy * targetDy);
            
            if (targetDistance > stabilization.movementThreshold) {
                stabilizationData.targetPositions[finger] = { x: currentX, y: currentY };
            }
        }
        
        // Calculate smooth transition
        const transitionTime = now - stabilizationData.transitionStartTime[finger];
        const maxTransitionTime = 1000 / stabilization.transitionSpeed; // Convert to milliseconds
        const progress = Math.min(transitionTime / maxTransitionTime, 1);
        
        // Apply easing function
        const easingFunc = this.easingFunctions[stabilization.easingType] || this.easingFunctions.easeOutCubic;
        const easedProgress = easingFunc(progress);
        
        // Interpolate between start and target position
        const startPos = stabilizationData.transitionStartPosition[finger];
        const targetPosCurrent = stabilizationData.targetPositions[finger];
        
        const interpolatedX = startPos.x + (targetPosCurrent.x - startPos.x) * easedProgress;
        const interpolatedY = startPos.y + (targetPosCurrent.y - startPos.y) * easedProgress;
        
        // Check if transition is complete
        if (progress >= 1) {
            stabilizationData.isTransitioning[finger] = false;
            stabilizationData.lastPositions[finger] = { x: targetPosCurrent.x, y: targetPosCurrent.y };
        }
        
        return { x: interpolatedX, y: interpolatedY };
    }
    
    // NEW: Calculate stabilized DIP position to prevent angle-based shaking
    calculateStabilizedDipPosition(dipLandmark, finger) {
        const currentX = dipLandmark.x * this.canvas.width;
        const currentY = dipLandmark.y * this.canvas.height;
        
        // If stabilization is disabled, return current position
        if (!this.settings.nailStabilization.enabled) {
            return { x: currentX, y: currentY };
        }
        
        const stabilization = this.settings.nailStabilization;
        const stabilizationData = this.nailStabilization;
        
        // Initialize DIP stabilization data for this finger if not exists
        if (!stabilizationData.lastDipPositions[finger]) {
            stabilizationData.lastDipPositions[finger] = { x: currentX, y: currentY };
            stabilizationData.targetDipPositions[finger] = { x: currentX, y: currentY };
            stabilizationData.isDipTransitioning[finger] = false;
            return { x: currentX, y: currentY };
        }
        
        const lastDipPos = stabilizationData.lastDipPositions[finger];
        const targetDipPos = stabilizationData.targetDipPositions[finger];
        
        // Calculate DIP movement distance
        const dx = currentX - lastDipPos.x;
        const dy = currentY - lastDipPos.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        
        // If DIP movement is below threshold, keep last DIP position (stabilize angle)
        if (distance < stabilization.movementThreshold) {
            return { x: lastDipPos.x, y: lastDipPos.y };
        }
        
        // DIP movement exceeds threshold - start or continue transition
        const now = Date.now();
        
        if (!stabilizationData.isDipTransitioning[finger]) {
            // Start new DIP transition
            stabilizationData.isDipTransitioning[finger] = true;
            stabilizationData.dipTransitionStartTime[finger] = now;
            stabilizationData.dipTransitionStartPosition[finger] = { x: lastDipPos.x, y: lastDipPos.y };
            stabilizationData.targetDipPositions[finger] = { x: currentX, y: currentY };
        } else {
            // Update target DIP position if movement is significant
            const targetDx = currentX - targetDipPos.x;
            const targetDy = currentY - targetDipPos.y;
            const targetDistance = Math.sqrt(targetDx * targetDx + targetDy * targetDy);
            
            if (targetDistance > stabilization.movementThreshold) {
                stabilizationData.targetDipPositions[finger] = { x: currentX, y: currentY };
            }
        }
        
        // Calculate smooth DIP transition
        const transitionTime = now - stabilizationData.dipTransitionStartTime[finger];
        const maxTransitionTime = 1000 / stabilization.transitionSpeed; // Convert to milliseconds
        const progress = Math.min(transitionTime / maxTransitionTime, 1);
        
        // Apply easing function
        const easingFunc = this.easingFunctions[stabilization.easingType] || this.easingFunctions.easeOutCubic;
        const easedProgress = easingFunc(progress);
        
        // Interpolate between start and target DIP position
        const startDipPos = stabilizationData.dipTransitionStartPosition[finger];
        const targetDipPosCurrent = stabilizationData.targetDipPositions[finger];
        
        const interpolatedX = startDipPos.x + (targetDipPosCurrent.x - startDipPos.x) * easedProgress;
        const interpolatedY = startDipPos.y + (targetDipPosCurrent.y - startDipPos.y) * easedProgress;
        
        // Check if DIP transition is complete
        if (progress >= 1) {
            stabilizationData.isDipTransitioning[finger] = false;
            stabilizationData.lastDipPositions[finger] = { x: targetDipPosCurrent.x, y: targetDipPosCurrent.y };
        }
        
        return { x: interpolatedX, y: interpolatedY };
    }
    
    // NEW: Calculate stabilized rotation to prevent angle-based shaking
    calculateStabilizedRotation(tipLandmark, dipLandmark, finger) {
        // Calculate current rotation angle
        const dx = tipLandmark.x - dipLandmark.x;
        const dy = tipLandmark.y - dipLandmark.y;
        const currentRotation = Math.atan2(dy, dx);
        
        // If rotation stabilization is disabled, return current rotation
        const rotationConfig = this.config.rotationStabilization;
        if (!rotationConfig?.enabled) {
            return currentRotation;
        }
        
        const stabilization = this.settings.nailStabilization;
        const stabilizationData = this.nailStabilization;
        
        // Initialize rotation history if not exists
        if (!stabilizationData.rotationHistory[finger]) {
            stabilizationData.rotationHistory[finger] = [];
        }
        
        // Add current rotation to history
        stabilizationData.rotationHistory[finger].push(currentRotation);
        if (stabilizationData.rotationHistory[finger].length > rotationConfig.historyLength) {
            stabilizationData.rotationHistory[finger].shift();
        }
        
        // Calculate average rotation from history for smoothing
        const rotationHistory = stabilizationData.rotationHistory[finger];
        const avgRotation = rotationHistory.reduce((sum, rot) => sum + rot, 0) / rotationHistory.length;
        
        // Initialize rotation stabilization data if not exists
        if (!stabilizationData.lastRotations[finger]) {
            stabilizationData.lastRotations[finger] = avgRotation;
            stabilizationData.targetRotations[finger] = avgRotation;
            stabilizationData.isRotationTransitioning[finger] = false;
            return avgRotation;
        }
        
        const lastRotation = stabilizationData.lastRotations[finger];
        const targetRotation = stabilizationData.targetRotations[finger];
        
        // Calculate rotation change
        let rotationChange = avgRotation - lastRotation;
        
        // Handle angle wrapping (e.g., from 3.1 to -3.1 radians)
        if (rotationChange > Math.PI) {
            rotationChange -= 2 * Math.PI;
        } else if (rotationChange < -Math.PI) {
            rotationChange += 2 * Math.PI;
        }
        
        const rotationThreshold = rotationConfig.threshold; // Use configured rotation threshold
        
        // If rotation change is below threshold, keep last rotation (stabilize angle)
        if (Math.abs(rotationChange) < rotationThreshold) {
            return lastRotation;
        }
        
        // Rotation change exceeds threshold - start or continue transition
        const now = Date.now();
        
        if (!stabilizationData.isRotationTransitioning[finger]) {
            // Start new rotation transition
            stabilizationData.isRotationTransitioning[finger] = true;
            stabilizationData.rotationTransitionStartTime[finger] = now;
            stabilizationData.rotationTransitionStartAngle[finger] = lastRotation;
            stabilizationData.targetRotations[finger] = avgRotation;
        } else {
            // Update target rotation if change is significant
            const targetRotationChange = avgRotation - targetRotation;
            if (Math.abs(targetRotationChange) > rotationThreshold) {
                stabilizationData.targetRotations[finger] = avgRotation;
            }
        }
        
        // Calculate smooth rotation transition
        const transitionTime = now - stabilizationData.rotationTransitionStartTime[finger];
        const maxTransitionTime = 1000 / stabilization.transitionSpeed; // Convert to milliseconds
        const progress = Math.min(transitionTime / maxTransitionTime, 1);
        
        // Apply easing function
        const easingFunc = this.easingFunctions[stabilization.easingType] || this.easingFunctions.easeOutCubic;
        const easedProgress = easingFunc(progress);
        
        // Interpolate between start and target rotation
        const startRotation = stabilizationData.rotationTransitionStartAngle[finger];
        const targetRotationCurrent = stabilizationData.targetRotations[finger];
        
        let interpolatedRotation = startRotation + (targetRotationCurrent - startRotation) * easedProgress;
        
        // Handle angle wrapping
        if (interpolatedRotation > Math.PI) {
            interpolatedRotation -= 2 * Math.PI;
        } else if (interpolatedRotation < -Math.PI) {
            interpolatedRotation += 2 * Math.PI;
        }
        
        // Check if rotation transition is complete
        if (progress >= 1) {
            stabilizationData.isRotationTransitioning[finger] = false;
            stabilizationData.lastRotations[finger] = targetRotationCurrent;
        }
        
        return interpolatedRotation;
    }
    
    drawNailAtLandmark(landmark, finger, dynamicScale = null, dynamicRotation = null) {
        // Calculate stabilized nail position
        const stabilizedPosition = this.calculateStabilizedNailPosition(landmark, finger);
        const x = stabilizedPosition.x;
        const y = stabilizedPosition.y;
        
        // Determine which scale to use
        let scale;
        if (this.settings.autoScale && dynamicScale !== null) {
            // Use dynamic scale when auto scale is enabled
            scale = dynamicScale;
        } else {
            // Use custom scale when auto scale is disabled, or fallback to base scale
            scale = this.settings.customScale * 0.3;
        }
        
        // Calculate final rotation: TIP-DIP angle + rotation offset
        const finalRotation = (dynamicRotation || 0) + this.settings.rotationOffset;
        
        // Draw the segmented nail image
        this.ctx.save();
        this.ctx.globalAlpha = this.settings.opacity;
        this.ctx.translate(x, y);
        this.ctx.rotate(finalRotation);
        
        const nailWidth = this.segmentedImage.width * scale;
        const nailHeight = this.segmentedImage.height * scale;
        
        this.ctx.drawImage(
            this.segmentedImage,
            -nailWidth / 2,
            -nailHeight / 2,
            nailWidth,
            nailHeight
        );
        
        this.ctx.restore();
    }
    
    calculateDynamicScaleAndRotation(tipLandmark, dipLandmark, pipLandmark, finger) {
      // Get stabilized DIP position to prevent angle-based shaking
      const stabilizedDipPos = this.calculateStabilizedDipPosition(
        dipLandmark,
        finger
      );

      // Convert stabilized DIP position back to normalized coordinates
      const stabilizedDipX = stabilizedDipPos.x / this.canvas.width;
      const stabilizedDipY = stabilizedDipPos.y / this.canvas.height;

      // Calculate distance between TIP and stabilized DIP
      const dx = tipLandmark.x - stabilizedDipX;
      const dy = tipLandmark.y - stabilizedDipY;
      const tipDipDistance = Math.sqrt(dx * dx + dy * dy);

      // Get finger-specific scaling parameters
      const fingerScaling =
        this.settings.fingerScaling[finger] ||
        this.settings.fingerScaling.index;

      // Calculate dynamic scale based on TIP-DIP distance and finger type
      let dynamicScale = fingerScaling.base;

      if (this.settings.autoScale) {
        // Normalize distance to scale range for this specific finger
        const normalizedDistance = Math.max(
          0,
          Math.min(
            1,
            (tipDipDistance - fingerScaling.min) /
              (fingerScaling.max - fingerScaling.min)
          )
        );

        // Scale from 0.1 to finger-specific max based on distance
        dynamicScale = 0.1 + normalizedDistance * (fingerScaling.base - 0.1);

        // Apply threshold - if distance is below threshold, use minimum scale
        if (tipDipDistance < this.settings.tipDipThreshold) {
          dynamicScale = 0.1; // Minimum scale for very close TIP-DIP
        }
      }

      // Calculate stabilized rotation to prevent angle-based shaking
      const rotation = this.calculateStabilizedRotation(
        tipLandmark,
        dipLandmark,
        finger
      );

      return {
        scale: dynamicScale,
        rotation: rotation,
      };
    }
    
    calculateFingerRotation(landmark) {
        // Fallback rotation calculation
        return (Math.random() - 0.5) * 0.2; // Small random rotation
    }
    
    // =============================================================================
    // NEW NAIL VISIBILITY DETECTION METHODS
    // =============================================================================
    
    /**
     * Main nail visibility detection method that uses configured approach
     */
    isNailVisible(tipLandmark, dipLandmark, pipLandmark, palmCenter, handedness) {
        const visibilityMethod = this.config.nailVisibility?.method || 'combined';
        
        switch (visibilityMethod) {
            case 'hand_confidence':
                return this.isNailVisibleByHandConfidence(tipLandmark, dipLandmark, pipLandmark, palmCenter, handedness);
            case 'landmark_stability':
                return this.isNailVisibleByLandmarkStability(tipLandmark, dipLandmark, pipLandmark, palmCenter, handedness);
            case 'custom_geometric':
                return this.isNailVisibleByCustomGeometric(tipLandmark, dipLandmark, pipLandmark, palmCenter, handedness);
            case 'combined':
            default:
                return this.isNailVisibleCombined(tipLandmark, dipLandmark, pipLandmark, palmCenter, handedness);
        }
    }
    
    /**
     * Method 1: Hand Detection Confidence
     * Uses MediaPipe's hand detection confidence score
     */
    isNailVisibleByHandConfidence(tipLandmark, dipLandmark, pipLandmark, palmCenter, handedness) {
        const handConfidenceConfig = this.config.nailVisibility?.handConfidence;
        if (!handConfidenceConfig?.enabled) return true;
        
        // Get hand confidence from current hand data
        const handConfidence = this.handData.confidence || 0;
        const threshold = handConfidenceConfig.threshold || 0.7;
        
        console.log(`Hand confidence: ${handConfidence}, threshold: ${threshold}`);
        return handConfidence >= threshold;
    }
    
    /**
     * Method 2: Landmark Position Stability
     * Uses landmark position stability over time
     */
    isNailVisibleByLandmarkStability(tipLandmark, dipLandmark, pipLandmark, palmCenter, handedness) {
        const stabilityConfig = this.config.nailVisibility?.landmarkStability;
        if (!stabilityConfig?.enabled) return true;
        
        // Check if TIP landmark is stable
        const tipIndex = this.TIP_DIP_MAP[this.getFingerFromLandmark(tipLandmark)]?.tip;
        if (tipIndex === undefined) return false;
        
        return this.isLandmarkStable(tipIndex);
    }
    
    /**
     * Method 3: Custom Geometric Logic (Enhanced)
     * Uses improved geometric calculations with configurable parameters
     */
    isNailVisibleByCustomGeometric(tipLandmark, dipLandmark, pipLandmark, palmCenter, handedness) {
        const geometricConfig = this.config.nailVisibility?.customGeometric;
        if (!geometricConfig?.enabled) return true;
        
        // Calculate finger direction vector (from DIP to TIP)
        const fingerDirection = {
            x: tipLandmark.x - dipLandmark.x,
            y: tipLandmark.y - dipLandmark.y
        };
        
        // Calculate palm-to-tip vector
        const palmToTip = {
            x: tipLandmark.x - palmCenter.x,
            y: tipLandmark.y - palmCenter.y
        };
        
        // Calculate distances for additional checks
        const fingerLength = Math.sqrt(fingerDirection.x * fingerDirection.x + fingerDirection.y * fingerDirection.y);
        const palmLength = Math.sqrt(palmToTip.x * palmToTip.x + palmToTip.y * palmToTip.y);
        
        if (fingerLength === 0 || palmLength === 0) return false;
        
        // Calculate angle between finger direction and palm-to-tip vector
        const dotProduct = fingerDirection.x * palmToTip.x + fingerDirection.y * palmToTip.y;
        const cosAngle = dotProduct / (fingerLength * palmLength);
        const angle = Math.acos(Math.max(-1, Math.min(1, cosAngle)));
        
        // Multiple geometric checks
        const angleThreshold = geometricConfig.angleThreshold || 0.5;
        const distanceThreshold = geometricConfig.distanceThreshold || 0.1;
        const visibilityThreshold = geometricConfig.visibilityThreshold || 0.3;
        
        // Check 1: Angle between finger and palm
        const angleCheck = angle < angleThreshold;
        
        // Check 2: Distance from palm (fingers too close to palm might be curled)
        const distanceCheck = palmLength > distanceThreshold;
        
        // Check 3: Original visibility threshold
        const visibilityCheck = angle < visibilityThreshold;
        
        console.log(`Geometric checks - angle: ${angle.toFixed(3)}, distance: ${palmLength.toFixed(3)}, angleCheck: ${angleCheck}, distanceCheck: ${distanceCheck}, visibilityCheck: ${visibilityCheck}`);
        
        return angleCheck && distanceCheck && visibilityCheck;
    }
    
    /**
     * Method 4: Combined Approach
     * Uses multiple methods with fallback logic
     */
    isNailVisibleCombined(tipLandmark, dipLandmark, pipLandmark, palmCenter, handedness) {
        const handConfidenceConfig = this.config.nailVisibility?.handConfidence;
        const stabilityConfig = this.config.nailVisibility?.landmarkStability;
        const geometricConfig = this.config.nailVisibility?.customGeometric;
        
        let results = {
            handConfidence: true,
            landmarkStability: true,
            customGeometric: true
        };
        
        // Check hand confidence if enabled
        if (handConfidenceConfig?.enabled) {
            const handConfidence = this.handData.confidence || 0;
            const threshold = handConfidenceConfig.threshold || 0.7;
            results.handConfidence = handConfidence >= threshold;
        }
        
        // Check landmark stability if enabled
        if (stabilityConfig?.enabled) {
            const tipIndex = this.TIP_DIP_MAP[this.getFingerFromLandmark(tipLandmark)]?.tip;
            results.landmarkStability = tipIndex !== undefined ? this.isLandmarkStable(tipIndex) : false;
        }
        
        // Check custom geometric if enabled
        if (geometricConfig?.enabled) {
            results.customGeometric = this.isNailVisibleByCustomGeometric(tipLandmark, dipLandmark, pipLandmark, palmCenter, handedness);
        }
        
        // Combined logic: at least one method must pass, or all enabled methods must pass
        const enabledMethods = [handConfidenceConfig?.enabled, stabilityConfig?.enabled, geometricConfig?.enabled].filter(Boolean).length;
        const passedMethods = Object.values(results).filter(Boolean).length;
        
        console.log(`Combined visibility check - enabled: ${enabledMethods}, passed: ${passedMethods}, results:`, results);
        
        // If only one method is enabled, it must pass
        if (enabledMethods === 1) {
            return passedMethods === 1;
        }
        
        // If multiple methods are enabled, at least one must pass
        return passedMethods > 0;
    }
    
    /**
     * Helper method to determine finger from landmark
     */
    getFingerFromLandmark(landmark) {
        // Use the current finger context set during processing
        return this.currentFinger || 'index';
    }
    
    // Check if landmark set is valid
    isValidLandmarkSet(landmarks, tipDipData) {
        return tipDipData.tip < landmarks.length && 
               tipDipData.dip < landmarks.length && 
               tipDipData.pip < landmarks.length &&
               this.PALM_CENTER < landmarks.length;
    }
    
    // Update landmark stability tracking for error handling
    updateLandmarkStability(landmarks) {
        const currentTime = Date.now();
        
        // Store current landmark positions
        this.landmarkHistory.positions.push({
            timestamp: currentTime,
            landmarks: landmarks.map(l => ({ x: l.x, y: l.y, z: l.z }))
        });
        
        // Keep only recent history
        if (this.landmarkHistory.positions.length > this.landmarkHistory.maxHistory) {
            this.landmarkHistory.positions.shift();
        }
        
        // Calculate stability for each landmark
        if (this.landmarkHistory.positions.length >= 3) {
            landmarks.forEach((landmark, index) => {
                const positions = this.landmarkHistory.positions.map(p => p.landmarks[index]);
                const variance = this.calculatePositionVariance(positions);
                this.landmarkHistory.stability[index] = variance;
            });
        }
    }
    
    // Calculate position variance for stability detection
    calculatePositionVariance(positions) {
        if (positions.length < 2) return 0;
        
        // Calculate mean position
        const meanX = positions.reduce((sum, p) => sum + p.x, 0) / positions.length;
        const meanY = positions.reduce((sum, p) => sum + p.y, 0) / positions.length;
        
        // Calculate variance
        const variance = positions.reduce((sum, p) => {
            const dx = p.x - meanX;
            const dy = p.y - meanY;
            return sum + (dx * dx + dy * dy);
        }, 0) / positions.length;
        
        return Math.sqrt(variance);
    }
    
    // Check if landmark is stable enough for rendering
    isLandmarkStable(landmarkIndex) {
        const stability = this.landmarkHistory.stability[landmarkIndex];
        return stability !== undefined && stability < this.settings.stabilityThreshold;
    }
    
    drawDebugInfo() {
      const ctx = this.ctx;
      ctx.fillStyle = "rgba(0, 0, 0, 0.8)";
      ctx.fillRect(10, 10, 350, 200);
      ctx.fillStyle = "rgba(255, 255, 255, 0.9)";
      ctx.font = "12px Arial";
      ctx.textAlign = "left";

      let y = 30;
      ctx.fillText(`Hands Detected: ${this.debugInfo.handCount}`, 20, y);
      y += 18;
      ctx.fillText(`FPS: ${this.debugInfo.fps}`, 20, y);
      y += 18;
      ctx.fillText(`Hand: ${this.handData.handedness || "None"}`, 20, y);
      y += 18;
      ctx.fillText(
        `Confidence: ${Math.round(this.handData.confidence * 100)}%`,
        20,
        y
      );
      y += 18;
      ctx.fillText(
        `Center: (${Math.round(this.handData.center.x)}, ${Math.round(
          this.handData.center.y
        )})`,
        20,
        y
      );
      y += 18;

      // Show visibility detection method
      const visibilityMethod = this.config.nailVisibility?.method || "combined";
      ctx.fillText(
        `Visibility Method: ${visibilityMethod.toUpperCase()}`,
        20,
        y
      );
      y += 18;

      // Show method-specific debug info
      if (this.config.nailVisibility?.handConfidence?.enabled) {
        const threshold = this.config.nailVisibility.handConfidence.threshold;
        ctx.fillText(
          `Hand Confidence: ${(this.handData.confidence * 100).toFixed(1)}% (${
            threshold * 100
          }%)`,
          20,
          y
        );
        y += 18;
      }

      if (this.config.nailVisibility?.landmarkStability?.enabled) {
        const threshold =
          this.config.nailVisibility.landmarkStability.threshold;
        ctx.fillText(`Stability Threshold: ${threshold}`, 20, y);
        y += 18;
      }

      if (this.config.nailVisibility?.customGeometric?.enabled) {
        const threshold =
          this.config.nailVisibility.customGeometric.visibilityThreshold;
        ctx.fillText(`Geometric Threshold: ${threshold}`, 20, y);
        y += 18;
      }

      // Show stabilization info
      if (this.settings.nailStabilization.enabled) {
        ctx.fillText(
          `Stabilization: ON (${this.settings.nailStabilization.movementThreshold}px)`,
          20,
          y
        );
        y += 18;

        const stabilizationStatus = this.getStabilizationStatus();
        const transitioningFingers = Object.keys(stabilizationStatus).filter(
          (finger) => stabilizationStatus[finger].isTransitioning
        );

        if (transitioningFingers.length > 0) {
          ctx.fillText(
            `TIP Transitioning: ${transitioningFingers.join(", ")}`,
            20,
            y
          );
          y += 18;
        }

        // Show DIP stabilization status
        const dipTransitioningFingers = Object.keys(
          this.nailStabilization.isDipTransitioning
        ).filter((finger) => this.nailStabilization.isDipTransitioning[finger]);

        if (dipTransitioningFingers.length > 0) {
          ctx.fillText(
            `DIP Transitioning: ${dipTransitioningFingers.join(", ")}`,
            20,
            y
          );
          y += 18;
        }

        // Show rotation stabilization status
        const rotationTransitioningFingers = Object.keys(
          this.nailStabilization.isRotationTransitioning
        ).filter(
          (finger) => this.nailStabilization.isRotationTransitioning[finger]
        );

        if (rotationTransitioningFingers.length > 0) {
          ctx.fillText(
            `Rotation Transitioning: ${rotationTransitioningFingers.join(
              ", "
            )}`,
            20,
            y
          );
          y += 18;
        }
      } else {
        ctx.fillText(`Stabilization: OFF`, 20, y);
        y += 18;
      }

      if (this.handData.isDetected) {
        ctx.strokeStyle = "rgba(0, 255, 0, 0.8)";
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.arc(
          this.handData.center.x,
          this.handData.center.y,
          10,
          0,
          2 * Math.PI
        );
        ctx.stroke();
      }
    }
    
    updateSettings(newSettings) {
        this.settings = { ...this.settings, ...newSettings };
        
        // Update nail stabilization settings if provided
        if (newSettings.nailStabilization) {
            this.settings.nailStabilization = { ...this.settings.nailStabilization, ...newSettings.nailStabilization };
        }
    }
    
    // Update nail stabilization settings
    updateNailStabilization(settings) {
        this.settings.nailStabilization = { ...this.settings.nailStabilization, ...settings };
    }
    
    // Get current stabilization status for debugging
    getStabilizationStatus() {
        const status = {};
        Object.keys(this.FINGER_MAP).forEach(finger => {
            const stabilizationData = this.nailStabilization;
            status[finger] = {
                isTransitioning: stabilizationData.isTransitioning[finger] || false,
                hasLastPosition: !!stabilizationData.lastPositions[finger],
                lastPosition: stabilizationData.lastPositions[finger] || null,
                targetPosition: stabilizationData.targetPositions[finger] || null
            };
        });
        return status;
    }
    
    // Reset stabilization data (useful for debugging)
    resetStabilization() {
        this.nailStabilization = {
            lastPositions: {},
            targetPositions: {},
            isTransitioning: {},
            transitionStartTime: {},
            transitionStartPosition: {},
            // Reset DIP stabilization data
            lastDipPositions: {},
            targetDipPositions: {},
            isDipTransitioning: {},
            dipTransitionStartTime: {},
            dipTransitionStartPosition: {}
        };
    }
    
    updateFingerSelection(finger, isSelected) {
        console.log(`Backend updateFingerSelection: ${finger} = ${isSelected}`);
        
        if (finger === 'all') {
            // When 'all' is toggled, update all fingers to match
            Object.keys(this.fingerSelection).forEach(key => {
                this.fingerSelection[key] = isSelected;
            });
            console.log('Updated all fingers to:', isSelected);
        } else {
            // Update the specific finger
            this.fingerSelection[finger] = isSelected;
            console.log(`Updated ${finger} to:`, isSelected);
        }
        
        console.log('Current finger selection state:', this.fingerSelection);
    }
    
    getHandData() {
        return {
            isDetected: this.handData.isDetected,
            confidence: this.handData.confidence,
            handedness: this.handData.handedness,
            distance: this.handData.distance,
            fps: this.debugInfo.fps,
            handCount: this.debugInfo.handCount,
            landmarks: this.handData.landmarks.length
        };
    }
}

// Make MediaPipeHandTracker available globally
window.MediaPipeHandTracker = MediaPipeHandTracker;
