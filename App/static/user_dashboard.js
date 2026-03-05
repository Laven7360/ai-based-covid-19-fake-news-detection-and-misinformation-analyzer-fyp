// static/js/user_dashboard.js

console.log("User dashboard JS loaded!");

function greetUser() {
    alert("Welcome, User! Here's your activity overview.");
    
    // Simulated data for now
    document.getElementById("userPosts").textContent = "12";
    document.getElementById("factChecks").textContent = "34";
    document.getElementById("reportsSent").textContent = "7";
}

window.onload = greetUser;