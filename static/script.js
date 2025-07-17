function loadFeature(feature) {
  document.getElementById("welcomePage").style.display = "none";
  document.getElementById("dashboard").style.display = "flex";
  document.querySelectorAll(".tab-content").forEach(el => el.classList.remove("active"));
  document.getElementById(feature).classList.add("active");
  document.querySelectorAll(".tab-button").forEach(el => {
    el.classList.toggle("active", el.dataset.target === feature);
  });
}

document.querySelectorAll('.tab-button').forEach(btn => {
  btn.addEventListener('click', () => {
    const target = btn.getAttribute('data-target');
    document.querySelectorAll('.tab-button').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));
    document.getElementById(target).classList.add('active');
  });
});

setInterval(() => {
  fetch('/latest_student')
    .then(res => res.json())
    .then(data => {
      document.getElementById('name').textContent = data.name;
      document.getElementById('id').textContent = data.id;
      document.getElementById('branch').textContent = data.branch;
    });
}, 2000);

function markPresent() {
  fetch('/mark_present', { method: 'POST' })
    .then(res => res.json())
    .then(data => {
      document.getElementById('presentStatus').textContent = data.message;
      setTimeout(() => document.getElementById('presentStatus').textContent = '', 3000);
    });
}

function getReportTable() {
  const start = document.getElementById("start").value;
  const end = document.getElementById("end").value;
  if (!start || !end) return alert("Select dates");
  fetch(`/get_attendance_data_range?start=${start}&end=${end}`)
    .then(res => res.json())
    .then(renderReportTable);
}

function getAllReports() {
  fetch(`/get_all_attendance`)
    .then(res => res.json())
    .then(renderReportTable);
}

function renderReportTable(data) {
  if (data.length === 0) {
    document.getElementById("reportResult").innerHTML = "<p>No data found.</p>";
    return;
  }

  const keys = Object.keys(data[0]);
  const fixed = ["Name", "StudentID", "Branch"];
  const dates = keys.filter(k => !fixed.includes(k)).sort();
  const ordered = [...fixed, ...dates];

  let html = '<table id="reportTable"><thead><tr><th>S.No</th>';
  ordered.forEach(k => html += `<th>${k}</th>`);
  html += '</tr></thead><tbody>';
  data.forEach((row, i) => {
    html += `<tr><td>${i + 1}</td>`;
    ordered.forEach(k => html += `<td>${row[k] || ''}</td>`);
    html += '</tr>';
  });
  html += '</tbody></table>';
  document.getElementById("reportResult").innerHTML = html;
}

function filterReport() {
  const val = document.getElementById("searchInput").value.toLowerCase();
  const nameIdx = [...document.querySelectorAll("#reportTable th")].findIndex(th => th.textContent === "Name");
  const idIdx = [...document.querySelectorAll("#reportTable th")].findIndex(th => th.textContent === "StudentID");
  document.querySelectorAll("#reportTable tbody tr").forEach(row => {
    const cells = row.querySelectorAll("td");
    const name = cells[nameIdx]?.textContent.toLowerCase() || "";
    const id = cells[idIdx]?.textContent.toLowerCase() || "";
    row.style.display = name.includes(val) || id.includes(val) ? "" : "none";
  });
}

function downloadReport() {
  const start = document.getElementById("start").value;
  const end = document.getElementById("end").value;
  if (!start || !end) return alert("Select dates");
  window.location.href = `/download_attendance_data_range?start=${start}&end=${end}`;
}

function calculatePercentage() {
  const id = document.getElementById("studentIDInput").value;
  if (!id) return alert("Enter student ID");
  fetch(`/calculate_attendance?student_id=${id}`)
    .then(res => res.json())
    .then(data => {
      document.getElementById("percentResult").textContent =
        data.error ? data.error : `Attendance: ${data.percentage}%`;
    });
}

function generateBatch() {
  const size = document.getElementById("batchSize").value;
  fetch(`/generate_batches?size=${size}`)
    .then(res => res.json())
    .then(data => {
      let html = "";
      data.forEach((batch, i) => {
        html += `<h3>Batch ${i + 1}</h3><ul>`;
        batch.forEach(s => {
          html += `<li>${s.StudentID} - ${s.Name} (${s.Branch})</li>`;
        });
        html += '</ul>';
      });
      document.getElementById("batchResult").innerHTML = html;
    });
}

function uploadNewData() {
  const csv = document.getElementById("csvFile").files[0];
  const zip = document.getElementById("zipFile").files[0];
  if (!csv || !zip) return alert("Both files required");
  const formData = new FormData();
  formData.append("csv_file", csv);
  formData.append("zip_file", zip);
  fetch('/upload_data', { method: 'POST', body: formData })
    .then(() => {
      document.getElementById("uploadStatus").textContent = `Uploaded: ${csv.name}, ${zip.name}`;
    })
    .catch(() => {
      document.getElementById("uploadStatus").textContent = "Upload failed.";
    });
}

window.onload = function () {
  const urlParams = new URLSearchParams(window.location.search);
  const tab = urlParams.get('tab') || 'attendance';
  loadFeature(tab);
}
function uploadMobileImage() {
  const file = document.getElementById("mobileImage").files[0];
  if (!file) return alert("Please select an image");

  const reader = new FileReader();
  reader.onloadend = function () {
    fetch("/recognize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: reader.result })
    })
      .then(res => res.json())
      .then(data => {
        document.getElementById("mobileResult").textContent = data.message;
      })
      .catch(() => {
        document.getElementById("mobileResult").textContent = "Error recognizing face.";
      });
  };
  reader.readAsDataURL(file);
};
