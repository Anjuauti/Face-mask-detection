// static/js/script.js
document.addEventListener('DOMContentLoaded', function() {
  // Main UI elements
  const mainContainer = document.getElementById('main-container');
  const imageOption = document.getElementById('image-option');
  const webcamOption = document.getElementById('webcam-option');
  const imageSection = document.getElementById('image-section');
  const webcamSection = document.getElementById('webcam-section');
  
  // Image upload elements
  const dropArea = document.getElementById('drop-area');
  const browseBtn = document.getElementById('browse-btn');
  const fileInput = document.getElementById('file-input');
  const processBtn = document.getElementById('process-btn');
  const originalImage = document.getElementById('original-image');
  const processedImage = document.getElementById('processed-image');
  const originalContainer = document.getElementById('original-container');
  const processedContainer = document.getElementById('processed-container');
  const downloadBtn = document.getElementById('download-btn');
  
  // Webcam elements
  const webcamStream = document.getElementById('webcam-stream');
  const webcamPlaceholder = document.getElementById('webcam-placeholder');
  const startCamera = document.getElementById('start-camera');
  const stopCamera = document.getElementById('stop-camera');
  
  // Back buttons
  const imageBackBtn = document.getElementById('image-back-btn');
  const webcamBackBtn = document.getElementById('webcam-back-btn');
  
  // Navigation
  imageOption.addEventListener('click', function() {
      mainContainer.style.display = 'none';
      imageSection.style.display = 'block';
  });
  
  webcamOption.addEventListener('click', function() {
      mainContainer.style.display = 'none';
      webcamSection.style.display = 'block';
  });
  
  imageBackBtn.addEventListener('click', function() {
      imageSection.style.display = 'none';
      mainContainer.style.display = 'block';
      resetImageSection();
  });
  
  webcamBackBtn.addEventListener('click', function() {
      webcamSection.style.display = 'none';
      mainContainer.style.display = 'block';
      stopWebcam();
  });
  
  // Reset image section
  function resetImageSection() {
      originalImage.src = '';
      processedImage.src = '';
      originalContainer.style.display = 'none';
      processedContainer.style.display = 'none';
      processBtn.disabled = true;
      window.selectedFile = null;
  }
  
  // Image upload functionality
  // Prevent default drag behaviors
  ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
      dropArea.addEventListener(eventName, preventDefaults, false);
  });
  
  function preventDefaults(e) {
      e.preventDefault();
      e.stopPropagation();
  }
  
  // Highlight drop area when dragging over it
  ['dragenter', 'dragover'].forEach(eventName => {
      dropArea.addEventListener(eventName, highlight, false);
  });
  
  ['dragleave', 'drop'].forEach(eventName => {
      dropArea.addEventListener(eventName, unhighlight, false);
  });
  
  function highlight() {
      dropArea.classList.add('active');
  }
  
  function unhighlight() {
      dropArea.classList.remove('active');
  }
  
  // Handle dropped files
  dropArea.addEventListener('drop', handleDrop, false);
  
  function handleDrop(e) {
      const dt = e.dataTransfer;
      const files = dt.files;
      handleFiles(files);
  }
  
  // Use the browse button to select files instead of clicking the entire drop area
  browseBtn.addEventListener('click', function(e) {
      e.stopPropagation(); // Prevent the click from triggering the dropArea click
      fileInput.click();
  });
  
  // Handle selected files from the file input
  fileInput.addEventListener('change', function() {
      handleFiles(this.files);
  });
  
  function handleFiles(files) {
      if (files.length > 0) {
          const file = files[0];
          
          // Check if file is an image
          if (!file.type.match('image.*')) {
              alert('Please select an image file');
              return;
          }
          
          // Display the selected image
          const reader = new FileReader();
          reader.onload = function(e) {
              originalImage.src = e.target.result;
              originalContainer.style.display = 'block';
              processedContainer.style.display = 'none';
              processBtn.disabled = false;
          };
          reader.readAsDataURL(file);
          
          // Save the file for processing
          window.selectedFile = file;
      }
  }
  
  // Process image button click
  processBtn.addEventListener('click', function() {
      if (!window.selectedFile) {
          alert('Please select an image first');
          return;
      }
      
      // Create FormData and append the file
      const formData = new FormData();
      formData.append('file', window.selectedFile);
      
      // Disable the button and show loading state
      processBtn.disabled = true;
      processBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
      
      // Send the file to the server for processing
      fetch('/upload', {
          method: 'POST',
          body: formData
      })
      .then(response => response.json())
      .then(data => {
          if (data.error) {
              alert(data.error);
          } else {
              // Add timestamp to prevent browser caching
              const timestamp = new Date().getTime();
              processedImage.src = data.processed + '?t=' + timestamp;
              processedContainer.style.display = 'block';
              
              // Update download link
              downloadBtn.href = data.processed;
          }
          
          // Reset button state
          processBtn.innerHTML = '<i class="fas fa-cogs"></i> Process Image';
          processBtn.disabled = false;
      })
      .catch(error => {
          console.error('Error:', error);
          alert('An error occurred during processing');
          
          // Reset button state
          processBtn.innerHTML = '<i class="fas fa-cogs"></i> Process Image';
          processBtn.disabled = false;
      });
  });
  
  // Webcam functionality
  // We don't need to redeclare webcamStream since it's already defined above
  
  startCamera.addEventListener('click', function() {
      // Start webcam only when user clicks the button
      startWebcam();
  });
  
  stopCamera.addEventListener('click', function() {
      stopWebcam();
  });
  
  function startWebcam() {
      startCamera.disabled = true;
      webcamPlaceholder.style.display = 'none';
      
      // Show loading state
      startCamera.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Starting...';
      
      // Tell the server to start the webcam
      fetch('/start_webcam', {
          method: 'POST'
      })
      .then(response => response.json())
      .then(data => {
          if (data.status === 'webcam started') {
              // Add a timestamp to prevent caching
              const timestamp = new Date().getTime();
              webcamStream.src = '/video_feed?t=' + timestamp;
              webcamStream.style.display = 'block';
              stopCamera.disabled = false;
              startCamera.innerHTML = '<i class="fas fa-play"></i> Start Camera';
          } else {
              throw new Error('Failed to start webcam');
          }
      })
      .catch(error => {
          console.error('Error starting webcam:', error);
          startCamera.disabled = false;
          startCamera.innerHTML = '<i class="fas fa-play"></i> Start Camera';
          alert('Failed to start webcam. Please check your camera permissions.');
      });
  }
  
  function stopWebcam() {
      webcamStream.style.display = 'none';
      webcamPlaceholder.style.display = 'flex';
      startCamera.disabled = false;
      stopCamera.disabled = true;
      
      // Tell the server to stop the webcam
      fetch('/stop_webcam', {
          method: 'POST'
      })
      .then(response => response.json())
      .catch(error => {
          console.error('Error stopping webcam:', error);
      });
      
      // Stop accessing the webcam by removing the src
      webcamStream.src = '';
  }
});