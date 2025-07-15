function Navbar() {
  return (
    <div>
      <nav class="navbar navbar-expand-lg " data-bs-theme="dark">
        <div class="container-fluid">
          <img src="../Images/main-logo.png" alt="Logo" />
          <button
            class="navbar-toggler"
            type="button"
            data-bs-toggle="collapse"
            data-bs-target="#navbarNavDropdown"
            aria-controls="navbarNavDropdown"
            aria-expanded="false"
            aria-label="Toggle navigation"
          >
            <span class="navbar-toggler-icon"></span>
          </button>
          <div class="collapse navbar-collapse" id="navbarNavDropdown">
            <ul class="navbar-nav">
              <li class="nav-item">
                <a class="nav-link active" aria-current="page" href="#">
                  Dashboard
                </a>
              </li>
              <li class="nav-item">
                <a class="nav-link active" href="#">
                  About us
                </a>
              </li>
              <li class="nav-item">
                <a class="nav-link active" href="#">
                  Contact us
                </a>
              </li>
            </ul>
          </div>
        </div>
        <button
          className="btn btn-danger"
          onClick={() => {
            window.location.href = "http://localhost:5000/";
          }}
        >
          Logout &nbsp;<i class="fa-solid fa-right-from-bracket"></i>
        </button>
      </nav>
      <div className="nav-container">
        <header className="header">
          <div className="greeting">
            <h1>Good afternoon, KrishiVikas Inc. 👋</h1>
            <p>Here is what’s happening with your projects today:</p>
          </div>
        </header>
      </div>
    </div>
  );
}

export default Navbar;
