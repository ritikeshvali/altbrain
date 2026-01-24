import React, { useState, useEffect } from 'react';

const API_BASE = 'http://localhost:8000';

function App() {
  const [collections, setCollections] = useState([]);
  const [selectedCollections, setSelectedCollections] = useState(['complete']);
  const [selectedFile, setSelectedFile] = useState(null);
  const [query, setQuery] = useState('');
  const [gdocLinks, setGdocLinks] = useState([]);
  const [currentGdocLink, setCurrentGdocLink] = useState('');
  const [response, setResponse] = useState('');
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('upload');
  const [uploadedFiles, setUploadedFiles] = useState({});

  useEffect(() => {
    fetchCollections();
  }, []);

  const fetchCollections = async () => {
    try {
      const res = await fetch(`${API_BASE}/collections`);
      const data = await res.json();
      setCollections(data.collections);
    } catch (err) {
      console.error(err);
    }
  };

  const fetchDocuments = async (collectionName) => {
    try {
      const res = await fetch(`${API_BASE}/documents/${collectionName}`);
      const data = await res.json();
      return data.documents || [];
    } catch (err) {
      return [];
    }
  };

  const loadAllDocuments = async () => {
    const allDocs = {};
    for (const coll of collections) {
      const docs = await fetchDocuments(coll.name);
      allDocs[coll.name] = docs;
    }
    setUploadedFiles(allDocs);
  };

  const deleteDocument = async (collection, filename) => {
    if (!window.confirm(`Delete ${filename}?`)) return;
    
    try {
      await fetch(`${API_BASE}/documents/${collection}?filename=${encodeURIComponent(filename)}`, {
        method: 'DELETE',
      });
      alert('Deleted');
      await loadAllDocuments();
      await fetchCollections();
    } catch (err) {
      alert('Delete failed: ' + err.message);
    }
  };

  useEffect(() => {
    if (activeTab === 'files') {
      loadAllDocuments();
    }
  }, [activeTab, collections]);

  const handleFileSelect = (e) => {
    setSelectedFile(e.target.files[0]);
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      alert('Select a file first');
      return;
    }
    if (selectedCollections.length === 0) {
      alert('Select at least one collection');
      return;
    }

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('collection', selectedCollections[0]);

    try {
      const res = await fetch(`${API_BASE}/ingest`, {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      alert(`✓ ${data.filename} (${data.chunks} chunks)`);
      
      setSelectedFile(null);
      document.getElementById('fileInput').value = '';
      await fetchCollections();
    } catch (err) {
      alert('Upload failed: ' + err.message);
    }
  };

  const addGdocLink = () => {
    if (currentGdocLink.trim()) {
      setGdocLinks([...gdocLinks, currentGdocLink.trim()]);
      setCurrentGdocLink('');
    }
  };

  const removeGdocLink = (index) => {
    setGdocLinks(gdocLinks.filter((_, i) => i !== index));
  };

  const handleAsk = async () => {
    if (!query.trim()) return;
    if (selectedCollections.length === 0) {
      alert('Select at least one collection');
      return;
    }

    setLoading(true);
    setResponse('');

    try {
      const res = await fetch(`${API_BASE}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          collections: selectedCollections,
          gdoc_link: gdocLinks.length > 0 ? gdocLinks[0] : null,
        }),
      });

      const reader = res.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value);
        setResponse(prev => prev + chunk);
      }
    } catch (err) {
      setResponse('AltBrain encountered an error: ' + err.message);
    } finally {
      setLoading(false);
      setQuery('');
      setGdocLinks([]);
    }
  };

  const toggleCollection = (name) => {
    setSelectedCollections(prev =>
      prev.includes(name) ? prev.filter(c => c !== name) : [...prev, name]
    );
  };

  const [embeddings, setEmbeddings] = useState({});
  const [showEmbeddings, setShowEmbeddings] = useState(false);

  const viewEmbeddings = async (collectionName) => {
    try {
      const res = await fetch(`${API_BASE}/embeddings/${collectionName}`);
      const data = await res.json();
      setEmbeddings(data);
      setShowEmbeddings(true);
    } catch (err) {
      alert('Error loading embeddings: ' + err.message);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <div className="max-w-6xl mx-auto">
        <h1 className="text-3xl font-bold mb-8 text-gray-800">
          AltBrain - Your Second Mind
        </h1>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Panel */}
          <div className="lg:col-span-1 space-y-6">
            {/* Tabs */}
            <div className="bg-white rounded-lg shadow">
              <div className="flex border-b">
                <button
                  onClick={() => setActiveTab('upload')}
                  className={`flex-1 py-3 font-medium ${
                    activeTab === 'upload'
                      ? 'text-blue-600 border-b-2 border-blue-600'
                      : 'text-gray-600'
                  }`}
                >
                  Upload
                </button>
                <button
                  onClick={() => setActiveTab('files')}
                  className={`flex-1 py-3 font-medium ${
                    activeTab === 'files'
                      ? 'text-blue-600 border-b-2 border-blue-600'
                      : 'text-gray-600'
                  }`}
                >
                  Files
                </button>
              </div>

              <div className="p-6">
                {activeTab === 'upload' ? (
                  <div>
                    <h2 className="text-lg font-semibold mb-4">Upload Document</h2>
                    <input
                      id="fileInput"
                      type="file"
                      accept=".pdf,.txt,.md,.doc,.docx,.jpg,.jpeg,.png"
                      onChange={handleFileSelect}
                      className="w-full text-sm mb-4 file:mr-4 file:py-2 file:px-4 
                        file:rounded file:border-0 file:bg-blue-50 file:text-blue-700 
                        hover:file:bg-blue-100 cursor-pointer"
                    />
                    {selectedFile && (
                      <p className="text-sm text-gray-600 mb-4">
                        Selected: {selectedFile.name}
                      </p>
                    )}
                    <button
                      onClick={handleUpload}
                      disabled={!selectedFile}
                      className="w-full py-2 bg-blue-600 text-white rounded-lg 
                        hover:bg-blue-700 disabled:bg-gray-300 font-medium"
                    >
                      Upload
                    </button>
                    <p className="text-xs text-gray-500 mt-2">
                      PDF, TXT, MD, DOC, DOCX, JPG, PNG
                    </p>
                  </div>
                ) : (
                  <div>
                    <h2 className="text-lg font-semibold mb-4">Uploaded Files</h2>
                    <div className="space-y-4 max-h-96 overflow-y-auto">
                      {collections.map(coll => (
                        <div key={coll.name}>
                          <h3 className="font-medium text-sm text-gray-700 mb-2">
                            {coll.name} ({coll.document_count})
                          </h3>
                          {uploadedFiles[coll.name]?.length > 0 ? (
                            <ul className="space-y-1 ml-2">
                              {uploadedFiles[coll.name].map((doc, idx) => (
                                <li key={idx} className="text-sm text-gray-600 flex justify-between items-center">
                                  <span>
                                    📄 {doc.filename}
                                    <span className="text-xs text-gray-400 ml-2">
                                      ({doc.chunks})
                                    </span>
                                  </span>
                                  <button
                                    onClick={() => deleteDocument(coll.name, doc.filename)}
                                    className="text-red-600 hover:text-red-800 text-xs"
                                  >
                                    Delete
                                  </button>
                                </li>
                              ))}
                            </ul>
                          ) : (
                            <p className="text-xs text-gray-400 ml-2">Empty</p>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Collections */}
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-lg font-semibold mb-4">Collections</h2>
              <div className="space-y-2">
                {collections.map(coll => (
                  <div key={coll.name} className="flex items-center justify-between">
                    <label className="flex items-center space-x-2 cursor-pointer flex-1">
                      <input
                        type="checkbox"
                        checked={selectedCollections.includes(coll.name)}
                        onChange={() => toggleCollection(coll.name)}
                        className="w-4 h-4 text-blue-600"
                      />
                      <div>
                        <div className="font-medium text-sm">{coll.name}</div>
                        <div className="text-xs text-gray-500">
                          {coll.document_count} docs
                        </div>
                      </div>
                    </label>
                    <button
                      onClick={() => viewEmbeddings(coll.name)}
                      className="ml-2 text-xs bg-purple-500 text-white px-2 py-1 rounded hover:bg-purple-600"
                    >
                      View Embeddings
                    </button>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Right Panel */}
          <div className="lg:col-span-2">
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-lg font-semibold mb-4">Ask Question</h2>

              <textarea
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Ask AltBrain info about yourself..."
                className="w-full h-24 p-3 border rounded-lg mb-4 
                  focus:outline-none focus:ring-2 focus:ring-blue-500"
              />

              {/* Google Doc Links */}
              <div className="mb-4">
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={currentGdocLink}
                    onChange={(e) => setCurrentGdocLink(e.target.value)}
                    placeholder="Google Doc link (optional)"
                    className="flex-1 p-3 border rounded-lg 
                      focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                  <button
                    onClick={addGdocLink}
                    className="px-4 py-2 bg-green-600 text-white rounded-lg 
                      hover:bg-green-700 font-medium"
                  >
                    Add
                  </button>
                </div>

                {gdocLinks.length > 0 && (
                  <div className="mt-3 space-y-2">
                    {gdocLinks.map((link, idx) => (
                      <div key={idx} className="flex items-center gap-2 text-sm">
                        <span className="flex-1 text-gray-600 truncate">
                          🔗 {link}
                        </span>
                        <button
                          onClick={() => removeGdocLink(idx)}
                          className="text-red-600 hover:text-red-800 text-xs"
                        >
                          Remove
                        </button>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              <button
                onClick={handleAsk}
                disabled={loading || !query.trim()}
                className="w-full py-3 bg-blue-600 text-white rounded-lg 
                  hover:bg-blue-700 disabled:bg-gray-300 font-medium"
              >
                {loading ? 'AltBrain is thinking...' : 'Ask'}
              </button>

              {response && (
                <div className="mt-6 p-4 bg-gray-50 rounded-lg">
                  <h3 className="font-semibold mb-2 text-gray-700">Answer from AltBrain:</h3>
                  <div className="prose prose-sm max-w-none whitespace-pre-wrap text-gray-800">
                    {response}
                  </div>
                </div>
              )}

              {showEmbeddings && (
                <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
                  <div className="bg-white rounded-lg p-6 max-w-2xl max-h-96 overflow-y-auto">
                    <div className="flex justify-between items-center mb-4">
                      <h2 className="text-xl font-bold">Embeddings: {embeddings.collection}</h2>
                      <button
                        onClick={() => setShowEmbeddings(false)}
                        className="text-gray-500 hover:text-gray-700 text-2xl"
                      >
                        ×
                      </button>
                    </div>
                    
                    <p className="text-sm text-gray-600 mb-4">
                      Total: {embeddings.total_embeddings} embeddings (1536 dimensions each)
                    </p>
                    
                    <div className="space-y-3">
                      {embeddings.embeddings?.map((emb, idx) => (
                        <div key={idx} className="border rounded p-3 bg-gray-50">
                          <p className="font-semibold text-sm">{idx + 1}. {emb.source}</p>
                          <p className="text-xs text-gray-600 mb-2">{emb.document_snippet}...</p>
                          <p className="text-xs font-mono bg-gray-100 p-2 rounded">
                            Embedding: [{emb.embedding_sample.map(x => x.toFixed(4)).join(', ')}...]
                          </p>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;