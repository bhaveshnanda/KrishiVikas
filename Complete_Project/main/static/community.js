document
  .getElementById("message-form")
  .addEventListener("submit", function (event) {
    event.preventDefault(); // Prevent the form from submitting

    // Get the input values
    const username = document.getElementById("username").value;
    const message = document.getElementById("message").value;

    // Create a new message element
    const messageItem = document.createElement("div");
    messageItem.classList.add("message-item");

    // Add the user's message and name
    messageItem.innerHTML = `<strong>${username}:</strong> <p>${message}</p>`;

    // Append the message to the discussion box
    document.getElementById("discussion-box").appendChild(messageItem);

    // Clear the input fields
    document.getElementById("message-form").reset();

    // Scroll to the bottom of the discussion box
    const discussionBox = document.getElementById("discussion-box");
    discussionBox.scrollTop = discussionBox.scrollHeight;
  });
